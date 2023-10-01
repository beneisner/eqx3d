import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp


class STN3d(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    bn1: nn.BatchNorm
    bn2: nn.BatchNorm
    bn3: nn.BatchNorm
    bn4: nn.BatchNorm
    bn5: nn.BatchNorm

    def __init__(self, channel, key):
        keys = jax.random.split(key, 6)

        self.conv1 = nn.Conv1d(channel, 64, 1, key=keys[0])
        self.conv2 = nn.Conv1d(64, 128, 1, key=keys[1])
        self.conv3 = nn.Conv1d(128, 1024, 1, key=keys[2])
        self.fc1 = nn.Linear(1024, 512, key=keys[3])
        self.fc2 = nn.Linear(512, 256, key=keys[4])
        self.fc3 = nn.Linear(256, 9, key=keys[5])

        self.bn1 = nn.BatchNorm(64, axis_name="batch")
        self.bn2 = nn.BatchNorm(128, axis_name="batch")
        self.bn3 = nn.BatchNorm(1024, axis_name="batch")
        self.bn4 = nn.BatchNorm(512, axis_name="batch")
        self.bn5 = nn.BatchNorm(256, axis_name="batch")

    def __call__(self, x, state):
        """Process a point cloud.

        Arguments:
            x: A JAX array of shape `(channel, num_points)`.
            state: A `State` object.

        Returns:
            A JAX array of shape `(3, 3)`.
        """

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x, state = self.bn3(x, state)
        x = jax.nn.relu(x)

        x = jnp.max(x, axis=-2, keepdims=False)

        x = self.fc1(x)
        x, state = self.bn4(x, state)
        x = jax.nn.relu(x)

        x = self.fc2(x)
        x, state = self.bn5(x, state)
        x = jax.nn.relu(x)

        x = self.fc3(x)

        iden = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32)

        x = x + iden
        x = x.reshape((3, 3))

        return x, state


class PointNetEncoder(eqx.Module):
    stn: STN3d
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d

    bn1: nn.BatchNorm
    bn2: nn.BatchNorm
    bn3: nn.BatchNorm

    def __init__(self, channel=3, key=0):
        keys = jax.random.split(key, 3)

        self.stn = STN3d(channel, keys[0])
        self.conv1 = nn.Conv1d(channel, 64, 1, key=keys[1])
        self.conv2 = nn.Conv1d(64, 128, 1, key=keys[2])
        self.conv3 = nn.Conv1d(128, 1024, 1, key=keys[3])

        self.bn1 = nn.BatchNorm(64, axis_name="batch")
        self.bn2 = nn.BatchNorm(128, axis_name="batch")
        self.bn3 = nn.BatchNorm(1024, axis_name="batch")

    def __call__(self, x, state):
        R, state = self.stn(x, state)

        # Apply the transformation to the input.
        x = R @ x

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x, state = self.bn3(x, state)

        x = jnp.max(x, -2, keepdims=False)

        return x, state


class PointNet(eqx.Module):
    feat: PointNetEncoder
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    dropout: nn.Dropout
    bn1: nn.BatchNorm
    bn2: nn.BatchNorm

    def __init__(self, k=40, key=None):
        keys = jax.random.split(key, 3)

        self.feat = PointNetEncoder(3, keys[0])
        self.fc1 = nn.Linear(1024, 512, key=keys[1])
        self.fc2 = nn.Linear(512, 256, key=keys[2])
        self.fc3 = nn.Linear(256, k, key=keys[3])

        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm(512, axis_name="batch")
        self.bn2 = nn.BatchNorm(256, axis_name="batch")

    def __call__(self, x, state, key):
        x, state = self.feat(x, state)

        x = self.fc1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.fc2(x)
        x = self.dropout(x, key=key)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)

        x = self.fc3(x)
        x = jax.nn.log_softmax(x, axis=-1)

        return x, state
