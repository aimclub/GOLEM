from golem.core.optimisers.optimization_parameters import GraphRequirements


class MolGraphRequirements(GraphRequirements):
    max_heavy_atoms: int = 50
