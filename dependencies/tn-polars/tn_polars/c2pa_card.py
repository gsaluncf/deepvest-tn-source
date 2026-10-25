"""A C2PA-signed release card: Content Credentials over the release evidence.

C2PA embeds manifests in supported media, and parquet is not one. The card is
a small PNG that quotes the release manifest's identifying facts, and the C2PA
manifest signed into the card carries those facts as an assertion. Verifying
the card therefore verifies a signed statement of what was released.
"""

import json

from PIL import Image, ImageDraw

ASSERTION_LABEL = "tn.polars.release"


def make_release_card(manifest, out_png):
    """Render the human-readable card the C2PA manifest gets embedded into."""
    image = Image.new("RGB", (640, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, 635, 215], outline=(23, 32, 51), width=2)
    lines = (
        "GovernedPolars release",
        f"policy: {manifest['policy_ref']}",
        f"file sha256: {manifest['file_sha256'][:32]}...",
        f"covered columns: {', '.join(manifest['covered_columns']) or 'none'}",
        f"label: {manifest['sensitivity_label']['name']}",
    )
    for index, line in enumerate(lines):
        draw.text((20, 24 + index * 34), line, fill=(23, 32, 51))
    image.save(out_png)
    return out_png


def release_assertion(manifest):
    """The claim the signed card makes about the exact release."""
    return {
        "label": ASSERTION_LABEL,
        "data": {
            "policy_ref": manifest["policy_ref"],
            "compiled_policy_sha256": manifest["compiled_policy_sha256"],
            "file_sha256": manifest["file_sha256"],
            "covered_columns": manifest["covered_columns"],
            "sensitivity_label": manifest["sensitivity_label"]["name"],
        },
    }


def sign_release_card(manifest, card_png, signed_png, *, cert_pem, key_pem,
                      alg="es256"):
    """Embed and sign a C2PA manifest carrying the release assertion."""
    from c2pa import Builder, C2paSignerInfo, Signer

    claim = {
        "claim_generator_info": [{"name": "tn-polars", "version": "0.1"}],
        "title": "GovernedPolars release card",
        "assertions": [release_assertion(manifest)],
    }
    info = C2paSignerInfo(alg=alg.encode(), sign_cert=cert_pem, private_key=key_pem,
                          ta_url=None)
    signer = Signer.from_info(info)
    builder = Builder(json.dumps(claim))
    with open(card_png, "rb") as source, open(signed_png, "wb") as dest:
        builder.sign(signer, "image/png", source, dest)
    return signed_png


def read_release_card(signed_png):
    """Return the release assertion data recovered from a signed card."""
    from c2pa import Reader

    with open(signed_png, "rb") as stream:
        reader = Reader("image/png", stream)
        store = json.loads(reader.json())
    manifest = store["manifests"][store["active_manifest"]]
    for assertion in manifest.get("assertions", []):
        if assertion.get("label") == ASSERTION_LABEL:
            return assertion["data"]
    raise ValueError("signed card carries no tn-polars release assertion")
