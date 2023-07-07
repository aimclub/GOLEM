import operator
from functools import reduce
from typing import Sequence, Optional, Any, Tuple, List, Iterable

import numpy as np

from golem.core.dag.graph import Graph
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
                 agent: OperatorAgent,
                 mutation_operator: Mutation,
                 objective: ObjectiveFunction):
        self.agent = agent
        self.mutation = mutation_operator
        self.objective = objective

    @property
    def _mutation_types(self) -> Sequence:
        # TODO: abstract that
        return self.agent.actions

    def fit(self, collector: HistoryCollector) -> OperatorAgent:
        for history in collector.load_histories():
            experience = ExperienceBuffer.from_history(history)
            # TODO: can I get oss/reward from there?
            self.agent.partial_fit(experience)
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

    def validate_history(self, history: OptHistory) -> Tuple[float, float]:
        """Validates history of mutated individuals against optimal policy."""
        history_trajectories = ExperienceBuffer.unroll_trajectories(history)
        return self._validate_against_optimal(history_trajectories)

    def validate_agent(self, graphs: Sequence[Graph]):
        """Validates agent policy against optimal policy on given graphs."""
        agent_steps = [self._make_action_step(Individual(g)) for g in graphs]
        return self._validate_against_optimal(trajectories=[agent_steps])

    def _validate_against_optimal(self, trajectories: Sequence[GraphTrajectory]) -> Tuple[float, float]:
        """Validates a policy trajectories against optimal policy
        that at each step always chooses the best action with max reward."""
        action_losses, reward_losses = [], []
        for trajectory in trajectories:
            inds, actions, rewards = unzip(trajectory)
            _, best_actions, best_rewards = unzip(map(self._apply_best_action, inds))
            action_loss, reward_loss = self._compute_loss(actions, rewards, best_actions, best_rewards)
            action_losses.append(action_loss)
            reward_losses.append(reward_loss)
        action_loss = float(np.mean(action_losses))
        reward_loss = float(np.mean(reward_losses))
        return action_loss, reward_loss

    def _compute_loss(self,
                      actions, rewards,
                      target_actions, target_rewards
                      ) -> Tuple[float, float]:
        # TODO: compute some loss! Either categorical OR reward loss OR both
        pass

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


def unzip(seq: Sequence[Tuple]) -> Tuple[Sequence]:
    return tuple(*zip(seq))


def concat_lists(lists: Iterable[List]) -> List:
    return reduce(operator.add, lists, [])
