"""Colab / Google-Drive repo-root bootstrap helpers.

Kept out of the notebook to reduce clutter. The notebook only needs enough inline
code to put this package on ``sys.path`` (trivial on a local kernel; on Colab it
mounts Drive first), then calls :func:`ensure_repo_root` / :func:`ensure_pkg_path`.
"""

from .bootstrap import ensure_pkg_path, ensure_repo_root, mount_drive

__all__ = ["ensure_pkg_path", "ensure_repo_root", "mount_drive"]
