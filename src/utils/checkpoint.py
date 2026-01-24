from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def save_model_weights(agent: nnx.Module, path: Path):
    _, state = nnx.split(agent)

    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path.absolute(), state)
