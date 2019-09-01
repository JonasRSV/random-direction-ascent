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

model.add(Layer(64, np.tanh))
env = gym.make("MountainCar-v0")

print(env.action_space)
print(env.action_space)

model.compile(env)
model.train(blackbox=play,
            env=env,
            iterations=1000,
            samples_per_update=1,
            postprocess=postprocess,
            show_after=True,
	    stop_at=-199)
