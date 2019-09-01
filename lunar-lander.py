import gym
import numpy as np
from gradient_free import (
    play,
    Model,
    Layer
)


def postprocess(x: np.ndarray):
    return np.argmax(x)


model = Model()

model.add(Layer(128, np.tanh))
env = gym.make("LunarLander-v2")

model.compile(env)
model.train(blackbox=play,
            env=env,
            iterations=2000,
            samples_per_update=2,
            postprocess=postprocess,
            show_after=True,
	    stop_at=200)
