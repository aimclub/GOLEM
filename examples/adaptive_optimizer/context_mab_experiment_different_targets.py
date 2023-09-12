from examples.adaptive_optimizer.experiment_setup import run_adaptive_mutations_with_context
from examples.adaptive_optimizer.mab_experiment_different_targets import run_experiment_node_num, \
    run_experiment_edge_num, run_experiment_graphs_ratio_edges_nodes, run_experiment_trees
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum


if __name__ == '__main__':
    """Run adaptive optimizer on different targets to see how neural multi-armed bandit agent converges
    to different probabilities of actions (i.e. mutations) for different targets."""
    adaptive_mutation_type = MutationAgentTypeEnum.contextual_bandit
    context_agent_type = ContextAgentTypeEnum.nodes_num

    run_experiment_node_num(trial_timeout=2,
                            adaptive_mutation_type=adaptive_mutation_type,
                            context_agent_type=context_agent_type,
                            run_func=run_adaptive_mutations_with_context)
    run_experiment_edge_num(trial_timeout=2,
                            adaptive_mutation_type=adaptive_mutation_type,
                            context_agent_type=context_agent_type,
                            run_func=run_adaptive_mutations_with_context)
    run_experiment_trees(trial_timeout=10, trial_iterations=2000,
                         adaptive_mutation_type=adaptive_mutation_type,
                         context_agent_type=context_agent_type,
                         run_func=run_adaptive_mutations_with_context)
    run_experiment_graphs_ratio_edges_nodes(trial_timeout=10, trial_iterations=2000,
                                            adaptive_mutation_type=adaptive_mutation_type,
                                            context_agent_type=context_agent_type,
                                            run_func=run_adaptive_mutations_with_context)
