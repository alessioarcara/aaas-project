import flax.nnx as nnx
import jax

from src.nets import CNN


def test_cnn():
    key = jax.random.key(0)
    net = CNN(
        obs_shape=(5, 2, 5, 4, 26),
        num_filters=[32],
        kernel_sizes=[3],
        strides=[1],
        paddings=["SAME"],
        dout=256,
        rngs=nnx.Rngs(0),
    )
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey, (5, 2, 5, 4, 26))

    out = net(x)

    assert out.shape == (5, 256)
