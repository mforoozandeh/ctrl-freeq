"""MkDocs hook: copy CHANGELOG.md into docs/ as release-notes.md at build time."""

import shutil
from pathlib import Path


def on_pre_build(config, **kwargs):
    repo_root = Path(config["config_file_path"]).parent
    src = repo_root / "CHANGELOG.md"
    dst = repo_root / "docs" / "release-notes.md"
    if src.exists():
        shutil.copy2(src, dst)
