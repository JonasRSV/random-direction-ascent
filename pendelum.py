import gym
import numpy as np
from gradient_free import (
    play,
    Model,
    Layer
)


def postprocess(x: np.ndarray):
    return np.tanh(x) * 2


model = Model()

model.add(Layer(64, np.tanh))
model.add(Layer(64, np.tanh))
model.add(Layer(64, np.tanh))

env = gym.make("Pendulum-v0")

model.compile(env)
model.train(blackbox=play,
            env=env,
            iterations=600,
            samples_per_update=2,
            postprocess=postprocess,
            show_after=True)
