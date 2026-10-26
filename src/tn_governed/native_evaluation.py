"""Execution checks derived from retained requests and native runtime events."""

from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request

from tn_llama_client.protocol import build_model_request
from tn_llama_client.transcript import HttpTranscript

from .native_reference import DESTINATION, MODEL_USE, REPORT_PROMPT
from .publications import write_once


def verify_admission_refusals(prepared, *, base_url: str, api_key: str):
    """Send matched invalid requests and require zero newly authorized model turns."""
    baseline = json.loads(
        build_model_request(
            prepared.calculation.calculation_object,
            REPORT_PROMPT,
            MODEL_USE,
            destination=DESTINATION,
        )
    )
    cases = []
    for field, value in [
        ("application", "unassigned.application"),
        ("operation", "unassigned.operation"),
    ]:
        request = copy.deepcopy(baseline)
        request["use"][field] = value
        cases.append(("wrong_" + field, request, "tn_governed_use_denied"))
    request = copy.deepcopy(baseline)
    request["destination"] = "unassigned.destination"
    cases.append(("wrong_destination", request, "tn_governed_use_denied"))
    request = copy.deepcopy(baseline)
    object = json.loads(request["tn_object"])
    signature = object["signature"]
    object["signature"] = ("A" if signature[0] != "A" else "B") + signature[1:]
    request["tn_object"] = json.dumps(object)
    cases.append(("invalid_signature", request, "tn_governed_object_invalid"))
    transcript = HttpTranscript(prepared.directory / "refusal-http.ndjson")
    audit = prepared.directory / "model-audit.ndjson"

    def turns():
        return sum(
            json.loads(line).get("kind") == "turn_begin_authorized"
            for line in audit.read_bytes().splitlines()
            if line
        )

    results = []
    for name, body, expected in cases:
        prior = turns()
        raw = json.dumps(body, separators=(",", ":")).encode()
        request = urllib.request.Request(
            base_url + "/v1/tn/chat/completions",
            data=raw,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        )
        transcript.log_request("llama", request, raw)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body, status = response.read(), response.status
                transcript.log_response("llama", request.full_url, response, response_body)
        except urllib.error.HTTPError as response:
            response_body, status = response.read(), response.code
            transcript.log_response("llama", request.full_url, response, response_body)
        code = json.loads(response_body).get("error", {}).get("code")
        after = turns()
        if not 400 <= status < 500 or code != expected or after != prior:
            raise AssertionError(f"{name} did not refuse before native model admission")
        results.append(
            {
                "case": name,
                "http_status": status,
                "error_code": code,
                "authorized_model_turns_before": prior,
                "authorized_model_turns_after": after,
            }
        )
    path = prepared.directory / "refusals.json"
    write_once(path, json.dumps(results, indent=2).encode())
    return path
