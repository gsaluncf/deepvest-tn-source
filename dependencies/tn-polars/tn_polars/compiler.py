import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from .policy import CompiledPolicy


_LEVELS = {"public", "internal", "confidential"}


def candidate_to_policy(candidate, tn_governance, declared_fields):
    if not isinstance(candidate, dict) or set(candidate) != {"schema", "policy_ref", "fields", "release"}:
        raise ValueError("unsupported policy candidate fields")
    if candidate.get("schema") != "tn.polars-policy-candidate.v1":
        raise ValueError("unexpected OpenRouter policy-candidate schema")
    if candidate.get("policy_ref") != tn_governance.get("policy"):
        raise ValueError("OpenRouter candidate is not bound to this TN policy reference")
    fields = candidate.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(declared_fields):
        raise ValueError("candidate fields must match the declared Polars input schema")
    if not all(isinstance(level, str) and level in _LEVELS for level in fields.values()):
        raise ValueError("candidate uses an unsupported field classification")
    return CompiledPolicy.from_dict({
        "schema": "tn.polars-policy.v1",
        "policy_ref": candidate["policy_ref"],
        "fields": fields,
        "release": candidate["release"],
    })


class OpenRouterPolicyCompiler:
    """Build-time translator from a TN governance body to a typed policy candidate."""

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    @staticmethod
    def candidate_schema(declared_fields):
        field_names = sorted(declared_fields)
        return {
            "name": "tn_polars_policy_candidate",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "schema": {"type": "string", "const": "tn.polars-policy-candidate.v1"},
                    "policy_ref": {"type": "string"},
                    "fields": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            name: {"type": "string", "enum": sorted(_LEVELS)}
                            for name in field_names
                        },
                        "required": field_names,
                    },
                    "release": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "public": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "allow_aggregates": {"type": "boolean"},
                                    "min_group_size": {"type": "integer", "minimum": 1},
                                },
                                "required": ["allow_aggregates", "min_group_size"],
                            }
                        },
                        "required": ["public"],
                    },
                },
                "required": ["schema", "policy_ref", "fields", "release"],
            },
        }

    def __init__(self, *, model=None, api_key=None):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.model = model or os.environ.get("TN_POLARS_OPENROUTER_MODEL", "tencent/hy3")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OpenRouter credential is unavailable for the tn-polars build step")

    def build_request(self, tn_governance, declared_fields, *, supporting_context=None):
        """Return the provider payload without credentials or an HTTP request."""
        schema = self.candidate_schema(declared_fields)
        prompt = {
            "task": "Compile this TN governance body into the constrained tn-polars policy candidate schema.",
            "constraints": [
                "Use only declared field names.",
                "Classify every declared field as public, internal, or confidential.",
                "Copy policy_ref exactly from governance.policy.",
                "Do not add prose or unsupported rules.",
            ],
            "governance": tn_governance,
            "declared_polars_fields": sorted(declared_fields),
            "supporting_context": supporting_context or {},
        }
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": json.dumps(prompt)}],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0,
        }

    def compile(self, tn_governance, declared_fields, *, supporting_context=None):
        payload = self.build_request(
            tn_governance, declared_fields, supporting_context=supporting_context,
        )
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json",
                     # Providers front this API with a WAF that rejects urllib's
                     # default signature (Cloudflare 1010) as a bare HTTP 403.
                     "User-Agent": "tn-polars-policy-compiler/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8"))
                message = detail.get("error", {}).get("message", "provider did not supply a message")
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                message = "provider did not supply a readable error message"
            raise RuntimeError(f"OpenRouter policy compilation failed: HTTP {error.code}: {message}") from error
        candidate = json.loads(result["choices"][0]["message"]["content"])
        return candidate_to_policy(candidate, tn_governance, declared_fields), candidate
