import io

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from ipywidgets import widgets
from PIL import Image

from src.base_agent import Agent
from src.config import Config
from src.overcooked_env import OvercookedGym
from src.utils.constants import TEST_LAYOUTS


def plot_test_results(layouts: list[str], mean_rewards: list[float], num_test_episodes: int) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(8, 6))

    sns.barplot(x=layouts, y=mean_rewards, hue=layouts, palette="viridis", legend=False, alpha=0.9)

    plt.xticks(ticks=range(len(layouts)), labels=layouts)
    plt.xticks(rotation=45, ha="right")

    plt.xlabel("Layout", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.title(
        f"Agent Pair Performance Across Different Layouts (Avg over {num_test_episodes} episodes)", fontsize=14, pad=20
    )

    for i, v in enumerate(mean_rewards):
        plt.text(i, v + 5, f"{v:.0f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show()


def show_overcooked_episode(cfg: Config, agent: Agent, seed: int) -> None:
    env = OvercookedGym(layouts=[TEST_LAYOUTS[0]], encoding=cfg.env_config.encoding)
    obs, _ = env.reset(seed=seed)
    done = False

    dd_layout = widgets.Dropdown(options=TEST_LAYOUTS, value=TEST_LAYOUTS[0], description="Map:")
    play = widgets.Play(min=0, max=cfg.env_config.horizon, interval=150)
    btn_reset = widgets.Button(description="Reset")
    img_widget = widgets.Image(format="png", width=500, height=500)

    def render():
        buf = io.BytesIO()
        Image.fromarray(env.render()).save(buf, format="PNG")
        img_widget.value = buf.getvalue()

    def run_step(change):
        nonlocal obs, done

        if done or change["new"] == 0:
            return

        obs_batch = jnp.stack(obs)[None]
        actions = np.array(agent.get_deterministic_action(obs_batch)).flatten()
        obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        render()

        if done:
            play.playing = False

    def reset_env(_=None):
        nonlocal obs, done
        play.playing = False
        play.value = 0
        obs, _ = env.reset(seed=seed)
        done = False
        render()

    def on_layout_change(change):
        nonlocal env
        play.playing = False
        env = OvercookedGym(layouts=[change["new"]], encoding=cfg.env_config.encoding)
        reset_env()

    play.observe(run_step, names="value")
    btn_reset.on_click(reset_env)
    dd_layout.observe(on_layout_change, names="value")

    display(widgets.VBox([dd_layout, widgets.HBox([play, btn_reset]), img_widget]))

    reset_env()
