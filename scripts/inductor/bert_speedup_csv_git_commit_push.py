#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Stage, commit, and push **only** the speedup CSV after a new row is appended.

Used by ``bert_multi_size_compile_debug_sweep.py`` when
``--git-commit-push-each-speedup-row`` is set. Can also be run directly::

    python scripts/inductor/bert_speedup_csv_git_commit_push.py \\
        --csv ./artifacts/speedup_by_config.csv \\
        --message "BERT speedup: base B=1 S=128"

Requirements: ``--csv`` must live inside a git work tree (``--repo-root`` can
override the inferred root). Ensure the CSV path is not gitignored if you need
it committed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional


def _git_top_level(start_dir: str) -> Optional[str]:
    start_dir = os.path.abspath(start_dir)
    try:
        out = subprocess.run(
            ["git", "-C", start_dir, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def commit_push_speedup_csv_only(
    csv_abspath: str,
    *,
    repo_root: Optional[str] = None,
    remote: str = "origin",
    branch: Optional[str] = None,
    commit_message: str,
) -> None:
    """``git add`` (single file), ``commit``, ``push`` — only that CSV path."""
    csv_abspath = os.path.abspath(csv_abspath)
    if not os.path.isfile(csv_abspath):
        raise FileNotFoundError(f"speedup CSV not found: {csv_abspath}")

    if repo_root is None:
        repo_root = _git_top_level(os.path.dirname(csv_abspath))
    else:
        repo_root = os.path.abspath(repo_root)

    if not repo_root:
        raise RuntimeError(
            f"Could not find a git repository containing {csv_abspath}. "
            "Pass --git-repo-root explicitly."
        )

    rel = os.path.relpath(csv_abspath, repo_root)
    if rel.startswith(".."):
        raise RuntimeError(
            f"CSV path {csv_abspath} is outside repo root {repo_root}"
        )

    subprocess.run(
        ["git", "-C", repo_root, "add", "--", rel],
        check=True,
    )
    diff = subprocess.run(
        ["git", "-C", repo_root, "diff", "--staged", "--quiet"],
    )
    if diff.returncode == 0:
        print(
            f"git: no staged changes for {rel} (nothing to commit; skipped).",
            file=sys.stderr,
            flush=True,
        )
        return

    subprocess.run(
        ["git", "-C", repo_root, "commit", "-m", commit_message],
        check=True,
    )

    if branch is None:
        br = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = br.stdout.strip()

    subprocess.run(
        ["git", "-C", repo_root, "push", remote, branch],
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Absolute or relative path to speedup_by_config.csv")
    p.add_argument("--repo-root", type=str, default=None, help="Git repo root (default: infer from CSV path)")
    p.add_argument("--remote", type=str, default="origin")
    p.add_argument("--branch", type=str, default=None, help="Branch to push (default: current HEAD)")
    p.add_argument("--message", "-m", type=str, required=True, help="Commit message")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        commit_push_speedup_csv_only(
            args.csv,
            repo_root=args.repo_root,
            remote=args.remote,
            branch=args.branch,
            commit_message=args.message,
        )
    except subprocess.CalledProcessError as e:
        print(f"git failed: {e}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
