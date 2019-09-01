import gym
import numpy as np
from gradient_free import (
    play,
    Model,
    Layer
)


def postprocess(x: np.ndarray):
    return np.tanh(x) * 2


def relu(x):
    return np.max(x, 0)

model = Model()

model.add(Layer(256, relu))
model.add(Layer(256, relu))

env = gym.make("Pendulum-v0")

model.compile(env)
model.train(blackbox=play,
            env=env,
            iterations=1000,
            samples_per_update=5,
            postprocess=postprocess,
            show_after=True,
	    stop_at=-200)
