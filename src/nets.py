from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(
        self,
        din: int,
        dhid: int,
        dout: int,
        rngs: nnx.Rngs,
    ):
        hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))

        self.lin1 = nnx.Linear(din, dhid, rngs=rngs, kernel_init=hidden_init)
        self.lin2 = nnx.Linear(dhid, dout, rngs=rngs, kernel_init=hidden_init)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.tanh(self.lin1(x))
        x = nnx.tanh(self.lin2(x))
        return x


class CNN(nnx.Module):
    def __init__(self, obs_shape: Sequence[int], num_filters: int, dout: int, rngs: nnx.Rngs):
        hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))

        self.conv1 = nnx.Conv(obs_shape[-1], num_filters, (3, 3), padding="SAME", rngs=rngs, kernel_init=hidden_init)
        self.conv2 = nnx.Conv(num_filters, num_filters, (3, 3), padding="SAME", rngs=rngs, kernel_init=hidden_init)
        self.conv3 = nnx.Conv(num_filters, num_filters, (3, 3), padding="SAME", rngs=rngs, kernel_init=hidden_init)

        self.flat_dim = obs_shape[-2] * obs_shape[-3] * num_filters  # H * W * C
        self.fc = nnx.Linear(self.flat_dim, dout, rngs=rngs, kernel_init=hidden_init)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))

        x = x.reshape(x.shape[0], -1)  # Flatten
        x = nnx.relu(self.fc(x))
        return x
