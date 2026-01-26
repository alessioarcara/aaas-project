# Autonomous and Adaptive Systems Project
alessio.arcara@studio.unibo.it

<div align="center">
  <a href="https://wandb.ai/aarcara/aaas_project/table?nw=nwuseralessioarcara"><img src="./assets/wandb_badge.svg" alt="W&B Report" style="height:28px; margin-top:0.75rem;"></a>
</div>

---

<h3 align="center">Learned Agents: Multi-layout Showcase</h3>

<table>
  <tr>
    <td colspan="2" width="33%">
      <img src="./assets/asymmetric_advantages.gif" width="100%" alt="Asymmetric Advantages">
      <br>
      <div align="center"><b>Asymmetric Advantages</b></div>
    </td>
    <td colspan="2" width="33%">
      <img src="./assets/coordination_ring.gif" width="100%" alt="Coordination Ring">
      <br>
      <div align="center"><b>Coordination Ring</b></div>
    </td>
    <td colspan="2" width="33%">
      <img src="./assets/counter_circuit.gif" width="100%" alt="Counter Circuit">
      <br>
      <div align="center"><b>Counter Circuit</b></div>
    </td>
  </tr>
  
  <tr>
    <td colspan="3" width="50%">
      <img src="./assets/cramped_room.gif" width="100%" alt="Cramped Room">
      <br>
      <div align="center"><b>Cramped Room</b></div>
    </td>
    <td colspan="3" width="50%">
      <img src="./assets/forced_coordination.gif" width="100%" alt="Forced Coordination">
      <br>
      <div align="center"><b>Forced Coordination</b></div>
    </td>
  </tr>
</table>

## Installation

<details>
<summary>Click to expand</summary>

### 1. **Clone the repository:**

```bash
git clone git@github.com:alessioarcara/aaas-project.git
cd aaas-project
```

### 2. **Set up the environment:**

You can set up the environment using `uv` or standard `pip`.

#### Option A: Using uv (recommended)

```bash
uv venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
uv sync
```

#### Option B: Using pip

This project adheres to PEP 621 standards using `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e .
```

</details>

## Usage

### Train an agent pair

This project uses configuration files to manage experiments (see `configs/`).
To train an agent pair, use the following script. Multiple configuration files can be stacked; the rightmost file overrides the previous ones.

```bash
uv run python scripts/train.py --configs configs/base.yaml configs/experiment_1.yaml
```

**Arguments:**
* `--configs`: One or more paths to YAML config files (space-separated).

### Test an agent pair

The `notebooks/testbench.ipynb` serves as an interactive tool for project analysis:

1. **Agent Pair Selection**: Select any agent pair from available checkpoints or from your own training runs using the checkpoint dropdown.
2. **Quantitative Metrics**: Automatically benchmark agents across layouts.
3. **Qualitative Assessment**: Visualize agent pair behavior in a selected layout using the layout dropdown.

### Tune hyperparameters

Hyperparameter optimization is performed using Optuna.

```bash
uv run tune.py --config configs/base.yaml --study-name your_study_name --trials 100
```
* --config: Base configuration file.
* --study-name: Name of the Optuna study.
* --trials: Number of optimization trials.

To modify which hyperparameters are optimized, edit the objective function inside the tuning script.


