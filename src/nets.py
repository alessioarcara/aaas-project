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


class CNN(nnx.Module):
    def __init__(self, obs_shape: int, num_filters: int, dout: int, rngs: nnx.Rngs):
        hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))
        bias_init = nnx.initializers.zeros

        self.conv1 = nnx.Conv(
            obs_shape[-1], num_filters, (3, 3), padding="SAME", rngs=rngs, kernel_init=hidden_init, bias_init=bias_init
        )
        self.conv2 = nnx.Conv(
            num_filters, num_filters, (3, 3), padding="SAME", rngs=rngs, kernel_init=hidden_init, bias_init=bias_init
        )
        self.conv3 = nnx.Conv(
            num_filters, num_filters, (3, 3), padding="SAME", rngs=rngs, kernel_init=hidden_init, bias_init=bias_init
        )

        self.flat_dim = obs_shape[-2] * obs_shape[-3] * num_filters  # Assuming input spatial dimensions are 5x4
        self.fc = nnx.Linear(self.flat_dim, dout, rngs=rngs, kernel_init=hidden_init, bias_init=bias_init)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, A, H, W, C = x.shape
        x = x.reshape((B * A, H, W, C))

        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))

        x = x.reshape(B * A, -1)  # (B*A, flat_dim)
        x = nnx.relu(self.fc(x))

        x = x.reshape(B, A, -1)  # (B, A, 256)

        return x
