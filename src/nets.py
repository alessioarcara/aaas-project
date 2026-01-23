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
    def __init__(
        self,
        obs_shape: Sequence[int],
        num_filters: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[str],
        dout: int,
        rngs: nnx.Rngs,
    ):
        hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))

        self.conv_layers = []
        cin = obs_shape[-1]

        for f, k, s, p in zip(num_filters, kernel_sizes, strides, paddings):
            self.conv_layers.append(
                nnx.Conv(cin, f, (k, k), strides=(s, s), padding=p, rngs=rngs, kernel_init=hidden_init)
            )
            cin = f

        # dummy forward to compute flat_dim
        dummy_input = jnp.zeros((1, *obs_shape[1:]))
        dummy_out = self._forward_conv(dummy_input)
        self.flat_dim = dummy_out.size

        self.fc = nnx.Linear(self.flat_dim, dout, rngs=rngs, kernel_init=hidden_init)

    def _forward_conv(self, x: jax.Array) -> jax.Array:
        for conv in self.conv_layers:
            x = nnx.relu(conv(x))
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self._forward_conv(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = nnx.relu(self.fc(x))
        return x
