from typing import Sequence, Optional, Any, Tuple

from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.history_collector import HistoryCollector
from golem.core.optimisers.adaptive.operator_agent import ExperienceBuffer, OperatorAgent
from golem.core.optimisers.fitness import null_fitness, Fitness
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


MutationIdType = Any


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
            self.agent.partial_fit(experience)
        return self.agent

    def validate_history(self, history: OptHistory) -> float:
        """Validates history of mutated individuals against ideal scenario
        when for each graph the best (by fitness) mutation is chosen."""
        experience = ExperienceBuffer.from_history(history)
        graphs, actual_mutations, rewards = experience.retrieve_experience()
        best_mutations, best_rewards = tuple(*zip(map(self._select_best_mutation, graphs)))
        categorical_loss, reward_loss = self._compute_loss(actual_mutations, rewards,
                                                           best_mutations, best_rewards)

    def validate_agent(self, graphs: Sequence[Graph]):
        # TODO: it is much more correct to estimate policy on rollouts,
        #  instead of per-point basis. That's why *history is a trajectory*,
        #  and here we collect *approximately same size of trajectory*.
        best_mutations, best_rewards = tuple(*zip(map(self._select_best_mutation, graphs)))
        actual_mutations, actual_rewards = [], []
        for graph in graphs:
            predicted_action = self.agent.choose_action(graph)
            reward = self._compute_reward(predicted_action, graph)
            actual_mutations.append(predicted_action)
            actual_rewards.append(reward)
        categorical_loss, reward_loss = self._compute_loss(actual_mutations, actual_rewards,
                                                           best_mutations, best_rewards)
        # TODO

    def _compute_loss(self,
                      actions, rewards,
                      target_actions, target_rewards
                      ) -> Tuple[float, float]:
        # TODO: compute some loss! Either categorical OR reward loss OR both
        pass

    def _select_best_mutation(self, graph: Graph) -> Tuple[MutationIdType, float]:
        """Returns best mutation for given graph and associated reward for choosing it."""
        prev_fitness = self.objective(graph)
        rewards = {mutation_id: self._compute_reward(mutation_id, graph, prev_fitness)
                   for mutation_id in self._mutation_types}
        best_mutation = max(rewards.keys(), key=lambda m: rewards[m])
        return best_mutation, rewards[best_mutation]

    def _compute_reward(self, action: MutationIdType, graph: Graph, prev_fitness: Optional[Fitness] = None) -> float:
        new_graph, applied = self.mutation._adapt_and_apply_mutation(graph, action)
        fitness = null_fitness() if applied else self.objective(new_graph)
        prev_fitness = prev_fitness or self.objective(graph)
        reward = prev_fitness.value - fitness.value if prev_fitness is not None else 0.
        return reward

