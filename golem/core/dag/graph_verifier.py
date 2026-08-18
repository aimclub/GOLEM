from typing import Sequence, Optional, Callable

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.adapter.adapter import IdentityAdapter, _transform
from golem.core.adapter.adapt_registry import AdaptRegistry
from golem.core.dag.graph import Graph
from golem.core.log import default_log

# Validation rule can either return False or raise a ValueError to signal a failed check
VerifierRuleType = Callable[..., bool]


class VerificationError(ValueError):
    pass


class GraphVerifier:
    def __init__(self, rules: Sequence[VerifierRuleType] = (),
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 raise_on_failure: bool = False):
        self._adapter = adapter or IdentityAdapter()
        self._rules = rules
        self._log = default_log(self)
        self._raise = raise_on_failure

    def __call__(self, graph: Graph) -> bool:
        return self.verify(graph)

    def verify(self, graph: Graph) -> bool:
        # Check if all rules pass.
        # The domain graph is restored at most once per verification: restoring
        # anew for every rule, as ``adapt_func`` would, dominates the cost of
        # verifying graphs whose domain representation is expensive to build.
        restore = _restore_memoized(self._adapter)
        for rule in self._rules:
            adapted_rule = rule if AdaptRegistry.is_native(rule) else                 _transform(rule, f_args=restore, f_ret=self._adapter.adapt)
            try:
                if adapted_rule(graph) is False:
                    return False
            except ValueError as err:
                msg = f'Graph verification failed with error <{err}> '\
                      f'for rule={rule} on graph={graph.descriptive_id}.'
                if self._raise:
                    raise VerificationError(msg)
                else:
                    self._log.debug(msg)
                    return False
        return True


def _restore_memoized(adapter: BaseOptimizationAdapter) -> Callable:
    """A ``restore`` that maps each object at most once, by identity.

    Verification rules only read the domain graph, so all rules of one
    verification can share a single restored instance.
    """
    memo = {}

    def restore(item):
        key = id(item)
        if key not in memo:
            memo[key] = adapter.restore(item)
        return memo[key]

    return restore
