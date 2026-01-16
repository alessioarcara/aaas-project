import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(
        self,
        din: int,
        dhid: int,
        dout: int,
        out_scale: float,
        rngs: nnx.Rngs,
    ):
        hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))
        bias_init = nnx.initializers.zeros
        out_init = nnx.initializers.orthogonal(out_scale)

        self.lin1 = nnx.Linear(din, dhid, rngs=rngs, kernel_init=hidden_init, bias_init=bias_init)
        self.lin2 = nnx.Linear(dhid, dhid, rngs=rngs, kernel_init=hidden_init, bias_init=bias_init)
        self.lin3 = nnx.Linear(dhid, dout, rngs=rngs, kernel_init=out_init, bias_init=bias_init)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.lin1(x)
        x = nnx.tanh(x)

        x = self.lin2(x)
        x = nnx.tanh(x)

        x = self.lin3(x)
        return x
