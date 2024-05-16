import gymnasium as gym
import numpy as np

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.parameters.parameter import AdaptiveParameter
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError


class GuidedAdaptiveParameter(AdaptiveParameter[int]):
    """
    Adaptive parameter that changes based on the performance of the optimizer.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def set(self, value):
        self.value = value

    @property
    def initial(self):
        return self.value

    def next(self, population):
        return self.value


class HyperparameterEnv(gym.Env):
    """
    Environment for hyperparameter optimization.
    Creates a graph optimizer at reset.

    Actions are choices of hyperparameters.
    Reward is derived from the optimizer's performance.
    """
    optimizer: EvoGraphOptimizer

    def __init__(self, optimizer_setup, objective, reward_coefficient=1.0):
        super().__init__()
        self.optimizer_setup = optimizer_setup
        self.objective = objective
        self.reward_coefficient = reward_coefficient

        # Define action space (continuous space for pop_size)
        self.action_space = gym.spaces.Box(low=np.log(5), high=np.log(200), shape=(1,))

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            'fitness_std': gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            'fitness_improvement': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'stagnation_counter': gym.spaces.Discrete(100)  # Assuming max stagnation of 100 generations
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a new AdaptiveParameter[int] instance for pop_size
        # TODO: don't start evolution?
        self.pop_size_adaptor = GuidedAdaptiveParameter(name='pop_size', value=np.exp(self.action_space.sample()))
        # Create a new optimizer instance with the pop_size_adaptor
        self.optimizer = self.optimizer_setup(pop_size_adaptor=self.pop_size_adaptor)

        self.best_fitness = -np.inf
        self.num_evals = 0
        evaluator = self.optimizer.eval_dispatcher.dispatch(self.objective, self.optimizer.timer)
        with self.optimizer.timer:
            self.optimizer._initial_population(evaluator)
        return self._get_observation(), {}

    def step(self, action):
        # Update pop_size based on the action
        pop_size = int(np.round(np.exp(action)))
        self.pop_size_adaptor.set(pop_size)

        # Run one generation of the optimizer
        terminated = self._run_iteration()

        # Calculate reward
        new_best_fitness = self.optimizer.history.archive_history[-1][0].fitness.value
        # For now, just use population size
        # TODO: compute the number of new individuals
        new_num_evals = pop_size
        reward = self.reward_coefficient * (new_best_fitness / self.best_fitness - 1) / (new_num_evals - self.num_evals)

        # Update state variables
        self.best_fitness = new_best_fitness
        self.num_evals = new_num_evals

        # Get observation
        observation = self._get_observation()

        return observation, reward, terminated, False, {}

    def _get_observation(self):
        if self.optimizer.generations.generation_num == 0:
            return {
                'fitness_std': np.array([0.0]),
                'fitness_improvement': np.array([0.0]),
                'stagnation_counter': 0
            }
        fitness_values = [ind.fitness.values[0] for ind in self.optimizer.population]
        fitness_std = np.std(fitness_values)
        fitness_improvement = self.optimizer.generations.is_quality_improved
        return {
            'fitness_std': np.array([fitness_std]),
            'fitness_improvement': np.array([fitness_improvement]),
            'stagnation_counter': self.optimizer.generations.stagnation_iter_count
        }

    def _run_iteration(self) -> bool:
        """ Runs one iteration of the optimizer.
            Returns if the optimization is done (termination condition met).
        """
        # Just like in PopulationalOptimizer, but without the progress bar and for one iteration

        with self.optimizer.timer:
            # TODO with self.timer: but how…
            if self.optimizer.stop_optimization():
                return True
            evaluator = self.optimizer.eval_dispatcher.dispatch(self.objective, self.optimizer.timer)
            try: 
                new_population = self.optimizer._evolve_population(evaluator)
                self.optimizer._update_population(new_population)
            except EvaluationAttemptsError as ex:
                self.optimizer.log.warning(f'Composition process was stopped due to: {ex}')
                return True
            return False
