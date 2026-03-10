#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _manifest(repo_root: Path) -> dict[str, object]:
    manifest_path = repo_root / "thirdparty" / "dsl_samples" / "MANIFEST.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _git_head(source_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prepare_source_tree(source_root: Path | None, *, repo_url: str, git_ref: str) -> tuple[Path, str]:
    if source_root is not None:
        return source_root.resolve(), _git_head(source_root.resolve())

    tempdir = Path(tempfile.mkdtemp(prefix="baybridge-thirdparty-"))
    subprocess.run(["git", "clone", repo_url, str(tempdir)], check=True)
    subprocess.run(["git", "checkout", git_ref], cwd=tempdir, check=True)
    return tempdir, _git_head(tempdir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the third-party DSL sample corpus")
    parser.add_argument("--source-root", type=Path, help="Existing upstream checkout root")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "thirdparty" / "dsl_samples",
        help="Output directory for generated third-party assets",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = _manifest(repo_root)
    repo_url = str(manifest["repo_url"])
    git_ref = str(manifest["git_ref"])
    selection = [str(item) for item in manifest["selection"]]
    output_root = args.output_root.resolve()
    upstream_root = output_root / "upstream"

    temp_clone: Path | None = None
    try:
        source_root, resolved_head = _prepare_source_tree(args.source_root, repo_url=repo_url, git_ref=git_ref)
        if args.source_root is None:
            temp_clone = source_root
        if upstream_root.exists():
            shutil.rmtree(upstream_root)
        upstream_root.mkdir(parents=True, exist_ok=True)

        for relative in selection:
            source = source_root / relative
            if not source.exists():
                raise FileNotFoundError(f"missing upstream file: {source}")
            destination = upstream_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        (output_root / "FETCHED.txt").write_text(
            f"repo_url={repo_url}\n"
            f"git_ref={git_ref}\n"
            f"resolved_head={resolved_head}\n"
            f"selection_count={len(selection)}\n",
            encoding="utf-8",
        )
        return 0
    finally:
        if temp_clone is not None:
            shutil.rmtree(temp_clone, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
