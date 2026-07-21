"""Rollout-video + inline-display helpers (extracted from the project notebook)."""

from __future__ import annotations

import base64
import math
import os
import random
from typing import Any, Callable

import imageio
import numpy as np


def video_path(name: str, folder_name: str = "videos") -> str:
    """Return a writable absolute path for a video file under ``folder_name/``."""
    folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def embed_mp4(filename: str):
    """Embed an mp4 file inline. Works in Colab and local Jupyter alike."""
    from IPython.display import HTML

    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    tag = (
        '<video width="640" height="480" controls>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
        "Your browser does not support the video tag."
        "</video>"
    )
    return HTML(tag)


def random_rollout_video(env, filename, max_steps=100, fps=10, seed=None):
    """Run one random-action episode and save it as mp4. Returns (steps, total_reward)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    with imageio.get_writer(filename, fps=fps) as video:
        video.append_data(env.render())
        for steps in range(1, max_steps + 1):
            action = random.randint(0, env.action_space.n - 1)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            video.append_data(env.render())
            if terminated or truncated:
                break
    return steps, total_reward


def agent_rollout_video(agent, env, filename, max_steps=100, fps=8, seed=None):
    """One greedy episode under ``agent``; save mp4 and return (steps, total_reward)."""
    obs, _ = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    with imageio.get_writer(filename, fps=fps) as video:
        video.append_data(env.render())
        for steps in range(1, max_steps + 1):
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            video.append_data(env.render())
            if terminated or truncated:
                break
    return steps, total_reward


def multi_rollout_video(
    agent,
    env,
    filename,
    *,
    n_episodes=4,
    max_steps=200,
    fps=8,
    seed=None,
    explore=False,
    gap_frames=4,
):
    """Record several episodes into a single mp4 for behavioural variety.

    Each episode uses a different seed (``seed + i``) so the map/spawn differ, giving
    a representative sample of the policy rather than one fixed trajectory. Episodes
    are separated by a short freeze on the final frame. ``explore=True`` uses the
    agent's current ε (stochastic) instead of greedy. Returns a list of
    ``(steps, total_reward)`` per episode.
    """
    results: list[tuple[int, float]] = []
    with imageio.get_writer(filename, fps=fps) as video:
        for i in range(int(n_episodes)):
            ep_seed = None if seed is None else int(seed) + i
            obs, _ = env.reset(seed=ep_seed)
            frame = env.render()
            video.append_data(frame)
            total_reward, steps = 0.0, 0
            for steps in range(1, max_steps + 1):
                action = agent.select_action(obs, explore=explore)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                frame = env.render()
                video.append_data(frame)
                if terminated or truncated:
                    break
            for _ in range(int(gap_frames)):  # brief freeze between episodes
                video.append_data(frame)
            results.append((steps, total_reward))
    return results


def grid_rollout_video(
    agent,
    env,
    filename,
    *,
    n_episodes=6,
    max_steps=200,
    fps=8,
    seed=None,
    explore=False,
    border=2,
    pad_color=0,
):
    """Record several episodes and tile them into a GRID that plays simultaneously.

    Instead of concatenating episodes end-to-end (hard to scrub through), every episode
    is rolled out, then all are shown side-by-side in one mp4 advancing in lock-step.
    Shorter episodes freeze on their final frame so every cell stays in sync. A near-
    square grid is used (e.g. 6 episodes -> 2x3). Returns a list of ``(steps, reward)``.
    """
    episodes: list[list[np.ndarray]] = []
    results: list[tuple[int, float]] = []
    for i in range(int(n_episodes)):
        ep_seed = None if seed is None else int(seed) + i
        obs, _ = env.reset(seed=ep_seed)
        frames = [np.asarray(env.render())]
        total_reward, steps = 0.0, 0
        for steps in range(1, int(max_steps) + 1):
            action = agent.select_action(obs, explore=explore)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            frames.append(np.asarray(env.render()))
            if terminated or truncated:
                break
        episodes.append(frames)
        results.append((steps, total_reward))

    # Pad every episode to the longest length by freezing on its last frame.
    length = max(len(f) for f in episodes)
    for frames in episodes:
        last = frames[-1]
        frames.extend([last] * (length - len(frames)))

    def _rgb(fr: np.ndarray) -> np.ndarray:
        return np.stack([fr] * 3, axis=-1) if fr.ndim == 2 else fr

    h, w = _rgb(episodes[0][0]).shape[:2]
    cols = int(math.ceil(math.sqrt(n_episodes)))
    rows = int(math.ceil(n_episodes / cols))
    canvas_h = rows * h + (rows + 1) * border
    canvas_w = cols * w + (cols + 1) * border

    with imageio.get_writer(filename, fps=fps) as video:
        for t in range(length):
            canvas = np.full((canvas_h, canvas_w, 3), pad_color, dtype=np.uint8)
            for idx in range(int(n_episodes)):
                r, c = divmod(idx, cols)
                y = border + r * (h + border)
                x = border + c * (w + border)
                canvas[y:y + h, x:x + w, :] = _rgb(episodes[idx][t])
            video.append_data(canvas)
    return results
