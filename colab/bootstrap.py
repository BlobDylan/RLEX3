"""Repo-root bootstrap for the notebook (local kernel or Colab / Google Drive).

``ensure_repo_root`` finds the folder holding ``algorithms/`` and ``wrappers/``,
puts it on ``sys.path`` and ``chdir``s into it. On Colab it mounts Drive first and
falls back to ``git clone``. ``ensure_pkg_path`` is a lighter re-pin for kernel
restarts / cwd drift.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Your project on Google Drive (after mount → /content/drive/MyDrive/...).
DRIVE_DIR = Path("/content/drive/MyDrive/RLEX3")
DRIVE_CANDIDATES = [
    DRIVE_DIR,
    Path("/content/drive/MyDrive/RLEX3"),
    Path("/content/drive/My Drive/RLEX3"),
    Path("/content/drive/MyDrive/University/Masters/Reinforcement Learning/Final Project/RLEX3"),
]

REPO_URL = os.environ.get("RLEX3_REPO_URL", "https://github.com/BlobDylan/RLEX3.git")
REPO_DIRNAME = "RLEX3"
BRANCH = os.environ.get("RLEX3_BRANCH", "feature/dqn-initial")
# Prefer Drive over git clone when both could work.
PREFER_DRIVE = True


def _has_packages(root: Path) -> bool:
    return (root / "algorithms").is_dir() and (root / "wrappers").is_dir()


def _activate(root: Path) -> Path:
    """Pin ``root`` at the front of ``sys.path`` and chdir into it."""
    root = root.resolve()
    rp = str(root)
    if rp in sys.path:
        sys.path.remove(rp)
    sys.path.insert(0, rp)
    os.chdir(root)
    return root


def mount_drive() -> bool:
    """Mount Google Drive so ``/content/drive/MyDrive/...`` is visible (Colab only)."""
    mount = Path("/content/drive")
    if mount.exists() and any(p.exists() for p in DRIVE_CANDIDATES):
        return True
    try:
        from google.colab import drive  # type: ignore
    except Exception as e:
        print(f"google.colab.drive not available ({e}); mount Drive from the Colab UI if needed.")
        return mount.exists()
    print("Mounting Google Drive …")
    try:
        drive.mount(str(mount), force_remount=False)
    except Exception as e:
        print(f"Drive mount failed: {e}")
        return False
    return mount.exists()


def _find_existing() -> Path | None:
    candidates: list[Path] = []
    if PREFER_DRIVE:
        candidates.extend(DRIVE_CANDIDATES)
    candidates.extend(
        [
            Path.cwd(),
            *Path.cwd().parents,
            Path.cwd() / REPO_DIRNAME,
            Path("/content") / REPO_DIRNAME,
            Path("/content") / "RLEX3",
        ]
    )
    if not PREFER_DRIVE:
        candidates.extend(DRIVE_CANDIDATES)
    seen: set[Path] = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if _has_packages(p):
            return p
    return None


def _try_clone() -> Path | None:
    dest = Path("/content") / REPO_DIRNAME
    if _has_packages(dest):
        return dest
    print(f"Cloning {REPO_URL} → {dest} …")
    try:
        subprocess.check_call(
            ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(dest)]
        )
    except subprocess.CalledProcessError:
        print("Branch clone failed; trying default branch…")
        if dest.exists():
            subprocess.check_call(["rm", "-rf", str(dest)])
        try:
            subprocess.check_call(["git", "clone", REPO_URL, str(dest)])
        except subprocess.CalledProcessError as e:
            print(f"git clone failed: {e}")
            return None
    return dest if _has_packages(dest) else None


def ensure_repo_root(verbose: bool = True) -> Path:
    """Find repo root (``algorithms/`` + ``wrappers/``), put on ``sys.path``, chdir in.

    Order: mount Drive on Colab → search Drive/local candidates → ``git clone``.
    """
    if Path("/content").is_dir():
        mount_drive()

    found = _find_existing()
    if found is None:
        found = _try_clone()
    if found is None:
        raise FileNotFoundError(
            "Could not find algorithms/ and wrappers/.\n"
            f"Expected Drive folder at: {DRIVE_DIR}\n"
            "Checklist:\n"
            "  1) Folder on Drive is named RLEX3 and contains algorithms/ and wrappers/\n"
            "  2) Re-run this cell and approve Drive access when prompted\n"
            "  3) Or set DRIVE_DIR to the exact path from: !ls /content/drive/MyDrive\n"
        )
    root = _activate(found)
    if verbose:
        print(f"repo root: {root}")
        print(f"cwd:       {Path.cwd()}")
        print(f"sys.path[0]: {sys.path[0]}")
        print("OK: algorithms/ and wrappers/ visible")
    return root


def ensure_pkg_path(verbose: bool = True) -> Path:
    """Re-pin the repo root on ``sys.path`` (defensive; kernel restart / cwd drift)."""
    found = _find_existing()
    if found is None:
        raise ModuleNotFoundError(
            "No module named 'algorithms'.\n"
            f"Looked for packages under {DRIVE_DIR}.\n"
            "Fix: run the Drive bootstrap cell, then re-run THIS cell.\n"
            "Check with:  import os; print(os.getcwd()); !ls algorithms"
        )
    root = _activate(found)
    if verbose:
        print(f"using repo root: {root}")
        print(f"cwd: {Path.cwd()}")
    return root
