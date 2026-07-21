"""Project helper utilities (device selection, training-curve plotting, rollout videos).

These are general-purpose helpers kept out of ``algorithms/`` (which holds the agents
themselves). Import from here in both the notebook and the scripts.
"""

from .device import describe_device, get_torch_device, seed_everything
from .plotting import (
    graphs_dir,
    load_training_history,
    pick_history,
    plot_training_history,
    rolling_mean,
    save_training_history,
)
from .video import (
    agent_rollout_video,
    embed_mp4,
    grid_rollout_video,
    multi_rollout_video,
    random_rollout_video,
    video_path,
)

__all__ = [
    "agent_rollout_video",
    "describe_device",
    "embed_mp4",
    "get_torch_device",
    "graphs_dir",
    "grid_rollout_video",
    "load_training_history",
    "multi_rollout_video",
    "pick_history",
    "plot_training_history",
    "random_rollout_video",
    "rolling_mean",
    "save_training_history",
    "seed_everything",
    "video_path",
]
