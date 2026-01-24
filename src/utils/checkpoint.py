from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def save_model_weights(agent: nnx.Module, path: Path):
    _, state = nnx.split(agent)

    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path.absolute(), state, force=True)


def load_model_weights(model_factory, path: Path) -> nnx.Module:
    abstract_model = nnx.eval_shape(model_factory)
    graphdef, abstract_state = nnx.split(abstract_model)

    with ocp.StandardCheckpointer() as ckptr:
        state_restored = ckptr.restore(path.absolute(), abstract_state)

    return nnx.merge(graphdef, state_restored)
