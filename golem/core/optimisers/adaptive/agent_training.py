from golem.core.optimisers.adaptive.history_collector import HistoryCollector
from golem.core.optimisers.adaptive.operator_agent import ExperienceBuffer, OperatorAgent


def fit_agent(collector: HistoryCollector, agent: OperatorAgent) -> OperatorAgent:
    for history in collector.load_histories():
        experience = ExperienceBuffer()
        experience.collect_history(history)
        agent.partial_fit(experience)
    return agent

