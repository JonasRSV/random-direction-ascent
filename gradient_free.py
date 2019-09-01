import sys
from typing import Callable, List
import numpy as np
import gym
from gym import spaces
import halo
import time
import seaborn as sb
import matplotlib.pyplot as plt


# cartpole-v1 mountaincar-v0 pendelum-v0

class Layer:

    def __init__(self, units: int, activation: Callable[[np.ndarray], np.ndarray]):
        self.units = units
        self.activation = activation

        self.W = None

        self.D = None

    def con(self, layer: 'Layer'):
        self.W = np.random.normal(0, 1 / layer.units, size=(layer.units, self.units))
        self.D = np.random.standard_normal(size=self.W.shape)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.W

    def update(self):
        self.W += self.D

    def reset(self):
        self.W -= self.D

    def explore(self):
        self.D = np.random.standard_normal(size=self.W.shape)


class Input:

    def __init__(self, units: int):
        self.units = units

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[-1] != self.units:
            raise ValueError("Input shape mismatch - got %s expected %s" % (inputs.shape[-1], self.units))

        return inputs

    def update(self):
        pass

    def reset(self):
        pass

    def explore(self):
        pass


Agent = Callable[[np.ndarray], np.ndarray]
Playable = Callable[[Agent, gym.Env, int, bool], float]


def play(entity: Agent,
         env: gym.Env,
         rounds: int,
         postprocess: Callable[[np.ndarray], np.ndarray],
         render: bool = False) -> float:
    fitness = 0
    for _ in range(rounds):
        s = env.reset()
        terminal = False
        while not terminal:
            if render:
                env.render()

            s, r, terminal, _ = env.step(postprocess(entity(s)))
            fitness += r

    return fitness


class Model:

    def __init__(self):
        self.layers: List[Layer] = []
        self.model = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, env: gym.Env):
        observation_space = env.observation_space.shape
        observation_space = 1 if not observation_space else observation_space[0]

        if isinstance(env.action_space, spaces.Discrete):
            action_space = env.action_space.n
        else:
            action_space = env.action_space.shape
            action_space = 1 if not action_space else action_space[0]

        self.model = [Input(observation_space)] + self.layers + [Layer(action_space, lambda x: x)]

        # Connect layers
        for i, l in enumerate(self.model[1:]):
            l.con(self.model[i])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        i = inputs
        for l in self.model:
            i = l.forward(i)

        return i

    @staticmethod
    def _spinner_text(progress: List[str], current_fitness: float, best_fitness: float) -> str:
        return f"[ {''.join(progress)} {' ' * (20 - len(progress))}  ] - best: {best_fitness} - current: {current_fitness} "

    def update(self):
        for l in self.model:
            l.update()

    def reset(self):
        for l in self.model:
            l.reset()

    def explore(self):
        for l in self.model:
            l.explore()

    def train(self, blackbox: Playable,
              env: gym.Env,
              iterations: int,
              samples_per_update: int,
              stop_at: float,
              show_after: bool,
              postprocess: Callable[[np.ndarray], np.ndarray]):
        timestamp = time.time()

        spinner = halo.Halo(text="Initialising full train", spinner="dots")
        spinner.start()

        fitnesses = []

        progress = []
        progress_symbol = "#"
        fitness = -1e6
        best = -1e6
        for i in range(iterations):
            spinner.text = self._spinner_text(progress, fitness, best)

            if 20 * i / iterations > len(progress):
                progress.append(progress_symbol)

            fitness = blackbox(self.predict, env, samples_per_update, postprocess) / samples_per_update

            if fitness > best:
                best = fitness
                self.update()

                if best >= stop_at:
                    break

            else:
                self.reset()
                self.explore()
                self.update()

            fitnesses.append(fitness)

        spinner.succeed(f"time: {time.time() - timestamp} seconds -- Final fitness {fitness} -- iterations {i}")

        self.reset()

        if show_after:
            fitness = blackbox(self.predict, env, 3, postprocess, True) / 3

            print("show got %.3f" % fitness)

            plt.figure(figsize=(20, 10))
            x = np.arange(len(fitnesses))
            sb.scatterplot(x, fitnesses)
            sb.lineplot(x, fitnesses)
            plt.show()
