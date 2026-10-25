"""Sensitivity labels for releases, in Microsoft Information Protection form.

The block always names a label derived from what the release carries. The
MSIP metadata keys are emitted only when the caller supplies their tenant's
label GUID, because fabricated GUIDs would look like a real tenant label.
"""

DEFAULT_LABEL_MAP = {
    "covered": {"name": "Confidential - Third-Party Licensed"},
    "clear": {"name": "General"},
}


def sensitivity_block(covered_columns, label_map=None):
    """Return the manifest block for a release with the given covered columns."""
    label_map = label_map or DEFAULT_LABEL_MAP
    entry = label_map["covered"] if covered_columns else label_map["clear"]
    block = {"name": entry["name"]}
    if entry.get("guid"):
        block["msip_metadata"] = msip_label_metadata(
            entry["name"], guid=entry["guid"], site_id=entry.get("site_id")
        )
    return block


def msip_label_metadata(name, *, guid, site_id=None, enabled=True):
    """MIP-compatible document metadata keys for one applied label."""
    prefix = f"MSIP_Label_{guid}_"
    metadata = {
        prefix + "Enabled": "true" if enabled else "false",
        prefix + "Name": name,
        prefix + "Method": "Privileged",
    }
    if site_id:
        metadata[prefix + "SiteId"] = site_id
    return metadata
