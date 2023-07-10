import operator
from functools import reduce
from typing import Sequence, Optional, Any, Tuple, List, Iterable

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.adaptive.history_collector import HistoryCollector
from golem.core.optimisers.adaptive.operator_agent import ExperienceBuffer, OperatorAgent, GraphTrajectory, \
    TrajectoryStep
from golem.core.optimisers.fitness import null_fitness, Fitness
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator

MutationIdType = Any
# Sequence of applied mutations and received rewards


class AgentLearner:
    def __init__(self,
                 objective: ObjectiveFunction,
                 mutation_operator: Mutation,
                 agent: Optional[OperatorAgent] = None,
                 ):
        self._log = default_log(self)
        self.agent = agent if agent is not None else mutation_operator.agent
        self.mutation = mutation_operator
        self.objective = objective

    @property
    def _mutation_types(self) -> Sequence:
        # TODO: abstract that
        return self.agent.actions

    def fit(self, collector: HistoryCollector, validate_each: int = -1) -> OperatorAgent:
        # TODO: how to understand that it learns at all?
        # -> check that reward on train trajectories increases at all
        # -> check that reward approaches theoretical maximum (optimal exhaustive policy)
        #   baseline <= agent <= optimal (pointwise)
        for i, history in enumerate(collector.load_histories()):
            experience = ExperienceBuffer.from_history(history)
            # TODO: can I get oss/reward from there?
            self.agent.partial_fit(experience)
            if validate_each > 0 and i % validate_each == 0:
                # TODO: get optimal rewards also
                reward_loss = self.validate_agent()

        return self.agent

    def validate_on_rollouts(self, histories: Sequence[OptHistory]) -> float:
        """Validates rollouts of agent vs. historic trajectories, comparing
        their mean total rewards (i.e. total fitness gain over the trajectory)."""

        # Collect all trajectories from all histories; and their rewards
        trajectories = concat_lists(map(ExperienceBuffer.unroll_trajectories, histories))

        mean_traj_len = int(np.mean([len(tr) for tr in trajectories]))
        traj_rewards = [sum(reward for _, reward, _ in traj) for traj in trajectories]
        mean_baseline_reward = np.mean(traj_rewards)

        # Collect same number of trajectories of the same length; and their rewards
        agent_trajectories = [self._sample_trajectory(initial=tr[0][0], length=mean_traj_len)
                              for tr in trajectories]
        agent_traj_rewards = [sum(reward for _, reward, _ in traj) for traj in agent_trajectories]
        mean_agent_reward = np.mean(agent_traj_rewards)

        # Compute improvement score of agent over baseline histories
        improvement = mean_agent_reward - mean_baseline_reward
        return improvement

    def validate_history(self, history: OptHistory) -> float:
        """Validates history of mutated individuals against optimal policy."""
        history_trajectories = ExperienceBuffer.unroll_trajectories(history)
        return self._validate_against_optimal(history_trajectories)

    def validate_agent(self,
                       graphs: Optional[Sequence[Graph]] = None,
                       history: Optional[OptHistory] = None) -> float:
        """Validates agent policy against optimal policy on given graphs."""
        if history is not None:
            agent_steps = ExperienceBuffer.from_history(history).retrieve_trajectories()
        elif graphs:
            agent_steps = [self._make_action_step(Individual(g)) for g in graphs]
        else:
            self._log.warning(f'Either graphs or history must not be None for validation!')
            return 0.
        return self._validate_against_optimal(trajectories=[agent_steps])

    def _validate_against_optimal(self, trajectories: Sequence[GraphTrajectory]) -> float:
        """Validates a policy trajectories against optimal policy
        that at each step always chooses the best action with max reward."""
        reward_losses = []
        for trajectory in trajectories:
            inds, actions, rewards = unzip(trajectory)
            _, best_actions, best_rewards = unzip(map(self._apply_best_action, inds))
            reward_loss = self._compute_reward_loss(rewards, best_rewards)
            reward_losses.append(reward_loss)
        reward_loss = float(np.mean(reward_losses))
        return reward_loss

    @staticmethod
    def _compute_reward_loss(rewards, optimal_rewards, normalized=False) -> float:
        """Returns difference (or deviation) from optimal reward.
        When normalized, 0. means actual rewards match optimal rewards completely,
        0.5 means they on average deviate by 50% from optimal rewards,
        and 2.2 means they on average deviate by more than 2 times from optimal reward."""
        reward_losses = np.subtract(optimal_rewards, rewards)  # always positive
        if normalized:
            reward_losses = reward_losses / np.abs(optimal_rewards)
        means = np.mean(reward_losses)
        return float(means)

    def _apply_best_action(self, graph: Graph) -> TrajectoryStep:
        """Returns greedily optimal mutation for given graph and associated reward."""
        ind = Individual(graph, fitness=self.objective(graph))
        results = {mutation_id: self._apply_action(mutation_id, ind)
                   for mutation_id in self._mutation_types}
        step = max(results.items(), key=lambda m: results[m][-1])
        return step

    def _apply_action(self, action: MutationIdType, ind: Individual) -> TrajectoryStep:
        new_graph, applied = self.mutation._adapt_and_apply_mutation(ind.graph, action)
        fitness = null_fitness() if applied else self.objective(new_graph)
        parent_op = ParentOperator(type_='mutation', operators=applied, parent_individuals=ind)
        new_ind = Individual(new_graph, fitness=fitness, parent_operator=parent_op)

        prev_fitness = ind.fitness or self.objective(ind.graph)
        reward = prev_fitness.value - fitness.value if prev_fitness is not None else 0.
        return new_ind, action, reward

    def _make_action_step(self, ind: Individual) -> TrajectoryStep:
        action = self.agent.choose_action(ind.graph)
        return self._apply_action(action, ind)

    def _sample_trajectory(self, initial: Individual, length: int) -> GraphTrajectory:
        trajectory = []
        past_ind = initial
        for i in range(length):
            next_ind, action, reward = self._make_action_step(past_ind)
            trajectory.append((next_ind, action, reward))
            past_ind = next_ind
        return trajectory


def unzip(seq: Iterable[Tuple]) -> Tuple[Sequence, ...]:
    return tuple(*zip(seq))


def concat_lists(lists: Iterable[List]) -> List:
    return reduce(operator.add, lists, [])
