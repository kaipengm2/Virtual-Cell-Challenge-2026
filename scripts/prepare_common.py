"""Historical asset verification shared by optional extraction scripts."""

import json
from pathlib import Path

from vcc_atlas.io import digest

MANIFEST = Path(__file__).resolve().parents[1] / "docs/raw-assets.json"


def verified_asset(name, path):
    record = json.loads(MANIFEST.read_text())[name]
    if Path(path).stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"{name}: raw asset differs from the historical source checksum")
    return record
