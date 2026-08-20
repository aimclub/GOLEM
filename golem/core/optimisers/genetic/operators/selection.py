import functools
import math
from copy import copy
from random import choice, randint, sample
from typing import Callable, List, Optional

from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.utilities.data_structures import ComparableEnum as Enum


class SelectionTypesEnum(Enum):
    tournament = 'tournament'
    spea2 = 'spea2'
    nsga2 = 'nsga2'
    tournament_with_replacement = 'tournament_with_replacement'


class Selection(Operator):
    """Selection operator.

    ``selection_types`` normally comes from the algorithm parameters; pass
    ``selection_types`` explicitly to build a selection that uses a different
    set -- used for the *mating* pool, whose requirements differ from those of
    environmental (survival) selection.
    """

    def __init__(self, parameters=None, requirements=None, selection_types=None):
        super().__init__(parameters=parameters, requirements=requirements)
        self._selection_types_override = selection_types

    @property
    def selection_types(self):
        return self._selection_types_override or self.parameters.selection_types

    def __call__(self, population: PopulationT, pop_size: Optional[int] = None) -> PopulationT:
        """
        Selection of individuals based on specified type of selection
        :param population: A list of individuals to select from.
        :param pop_size: Optional custom population_size.
        Taken from algorithm parameters if not specified.
        """
        pop_size = pop_size if pop_size is not None else self.parameters.pop_size
        selection_type = choice(self.selection_types)
        return self._selection_by_type(selection_type)(population, pop_size)

    @staticmethod
    def _selection_by_type(selection_type: SelectionTypesEnum) -> Callable[[PopulationT, int], PopulationT]:
        selections = {
            SelectionTypesEnum.tournament: tournament_selection,
            SelectionTypesEnum.spea2: spea2_selection,
            SelectionTypesEnum.nsga2: nsga2_selection,
            SelectionTypesEnum.tournament_with_replacement: tournament_selection_with_replacement,
        }
        if selection_type in selections:
            return selections[selection_type]
        elif isinstance(selection_type, Callable):
            return selection_type
        else:
            raise ValueError(f'Required selection not found: {selection_type}')


def default_selection_behaviour(selection_func: Optional[Callable] = None, *, ensure_unique: bool = True,
                                populate_by_single: bool = True):
    def func_wrapper(func: Callable):
        @functools.wraps(func)
        def wrapper(individuals: PopulationT, pop_size: int, *args, **kwargs):
            if ensure_unique:
                individuals = list({ind.uid: ind for ind in individuals}.values())
            else:
                individuals = copy(individuals)

            if populate_by_single and len(individuals) == 1:
                return individuals * pop_size

            if len(individuals) <= pop_size:
                return individuals

            return func(individuals, pop_size, *args, **kwargs)
        return wrapper

    if selection_func:
        return func_wrapper(selection_func)  # Allows to decorate without args.
    return func_wrapper  # Allows to decorate with args but no selection_func specified.


@default_selection_behaviour
def tournament_selection(individuals: PopulationT, pop_size: int, fraction: float = 0.1) -> PopulationT:
    """ Having the size of *individuals* equals to *n* will have no effect other
    than lax ordering of *individuals*. """
    individuals = list(individuals)  # don't modify original
    group_size = math.ceil(len(individuals) * fraction)
    min_group_size = min(2, len(individuals))
    group_size = max(group_size, min_group_size)
    chosen = []
    iterations_limit = pop_size * 10
    for _ in range(iterations_limit):
        if len(chosen) >= pop_size:
            break
        group = sample(individuals, min(group_size, len(individuals)))
        best = max(group, key=lambda ind: ind.fitness)
        individuals.remove(best)
        chosen.append(best)
    return chosen


@default_selection_behaviour
def random_selection(individuals: PopulationT, pop_size: int) -> PopulationT:
    return sample(individuals, pop_size)


def _tournament_winner(first: Individual, second: Individual) -> Individual:
    """Binary-tournament comparison that is correct for both fitness kinds.

    ``dominates`` is Pareto domination for multi-objective fitness and plain
    better-than for single-objective; mutually non-dominated pairs are decided
    by a coin flip, as in NSGA-II's binary tournament.
    """
    if first.fitness.dominates(second.fitness):
        return first
    if second.fitness.dominates(first.fitness):
        return second
    return choice((first, second))


def tournament_selection_with_replacement(individuals: PopulationT, pop_size: int,
                                          tournament_size: int = 2) -> PopulationT:
    """Build a *mating pool* of exactly ``pop_size`` parents, with replacement.

    Deliberately NOT wrapped in ``default_selection_behaviour``: that wrapper
    returns the input untouched whenever ``len(individuals) <= pop_size``,
    which is precisely the case in reproduction -- the mating pool is never
    smaller than the population it is drawn from. The result is that a
    generational run applies no mating pressure at all: every parent breeds
    regardless of fitness.

    Each tournament draws its contenders without replacement, so an
    individual never competes against itself; the tournaments themselves are
    independent, so a fit individual can win several and enter the pool more
    than once. That is what decouples the pool size from the population size
    and lets fitness decide how often a parent breeds.
    """
    individuals = list({ind.uid: ind for ind in individuals}.values())
    if not individuals:
        return []
    if len(individuals) == 1:
        return individuals * pop_size
    size = max(2, min(tournament_size, len(individuals)))
    chosen = []
    for _ in range(pop_size):
        group = sample(individuals, size)
        winner = group[0]
        for contender in group[1:]:
            winner = _tournament_winner(winner, contender)
        chosen.append(winner)
    return chosen


def fast_non_dominated_sort(individuals: PopulationT) -> List[List[Individual]]:
    """Partition ``individuals`` into Pareto fronts (NSGA-II, Deb et al. 2002)."""
    size = len(individuals)
    dominated_by = [[] for _ in range(size)]
    domination_count = [0] * size

    for i in range(size):
        for j in range(i + 1, size):
            if individuals[i].fitness.dominates(individuals[j].fitness):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif individuals[j].fitness.dominates(individuals[i].fitness):
                dominated_by[j].append(i)
                domination_count[i] += 1

    # A domination count is only final once every pair has been visited, so
    # the first front is read off after the loop, not during it.
    fronts: List[List[int]] = [[i for i in range(size) if domination_count[i] == 0]]
    current = fronts[0]
    while current:
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        current = nxt
    return [[individuals[i] for i in front] for front in fronts]


def crowding_distances(front: PopulationT) -> List[float]:
    """NSGA-II crowding distance: how isolated each solution is on its front."""
    size = len(front)
    if size <= 2:
        return [float('inf')] * size
    distances = [0.0] * size
    num_objectives = len(front[0].fitness.values)
    for axis in range(num_objectives):
        order = sorted(range(size), key=lambda i: front[i].fitness.values[axis])
        low = front[order[0]].fitness.values[axis]
        high = front[order[-1]].fitness.values[axis]
        distances[order[0]] = distances[order[-1]] = float('inf')
        span = high - low
        if span <= 0:
            continue
        for rank in range(1, size - 1):
            nxt = front[order[rank + 1]].fitness.values[axis]
            prev = front[order[rank - 1]].fitness.values[axis]
            distances[order[rank]] += (nxt - prev) / span
    return distances


@default_selection_behaviour
def nsga2_selection(individuals: PopulationT, pop_size: int) -> PopulationT:
    """NSGA-II environmental selection: Pareto rank, then crowding distance.

    An alternative to ``spea2_selection`` for multi-objective runs. SPEA-2
    scores density with a k-th-nearest-neighbour estimate over the whole
    population and truncates one point at a time -- O(N^3) in the worst case
    and biased towards whichever region happens to be dense. NSGA-II keeps
    whole fronts and only needs crowding distance on the front that
    overflows, which is O(M N log N) and spreads the retained solutions more
    evenly along the front -- what matters on objectives that are strongly
    correlated, where the front is a thin arc and losing its parsimonious end
    means losing the interpretable solutions.
    """
    chosen: List[Individual] = []
    for front in fast_non_dominated_sort(list(individuals)):
        if len(chosen) + len(front) <= pop_size:
            chosen.extend(front)
            continue
        distances = crowding_distances(front)
        order = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
        chosen.extend(front[i] for i in order[:pop_size - len(chosen)])
        break
    return chosen


# Code of spea2 selection is modified part of DEAP library (Library URL: https://github.com/DEAP/deap).
@default_selection_behaviour
def spea2_selection(individuals: PopulationT, pop_size: int) -> PopulationT:
    """
    Apply SPEA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *n* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *n* will have no effect other
    than sorting the population according to a strength Pareto scheme. The
    list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :returns: A list of selected individuals
    """
    inds_len = len(individuals)
    fitness_len = len(individuals[0].fitness.values)
    inds_len_sqrt = math.sqrt(inds_len)
    strength_fits = [0] * inds_len
    fits = [0] * inds_len
    dominating_inds = [list() for _ in range(inds_len)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1:], i + 1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    for i in range(inds_len):
        for j in dominating_inds[i]:
            fits[i] += strength_fits[j]

    # Choose all non-dominated individuals
    chosen_indices = [i for i in range(inds_len) if fits[i] < 1]

    if len(chosen_indices) < pop_size:  # The archive is too small
        for i in range(inds_len):
            distances = [0.0] * inds_len
            for j in range(i + 1, inds_len):
                dist = 0.0
                for idx in range(fitness_len):
                    val = \
                        individuals[i].fitness.values[idx] - \
                        individuals[j].fitness.values[idx]
                    dist += val * val
                distances[j] = dist
            kth_dist = _randomized_select(distances, 0, inds_len - 1, inds_len_sqrt)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density

        next_indices = [(fits[i], i) for i in range(inds_len)
                        if i not in chosen_indices]
        next_indices.sort()
        # print next_indices
        chosen_indices += [i for _, i in next_indices[:pop_size - len(chosen_indices)]]

    elif len(chosen_indices) > pop_size:  # The archive is too large
        inds_len = len(chosen_indices)
        distances = [[0.0] * inds_len for _ in range(inds_len)]
        sorted_indices = [[0] * inds_len for _ in range(inds_len)]
        for i in range(inds_len):
            for j in range(i + 1, inds_len):
                dist = 0.0
                for idx in range(fitness_len):
                    val = \
                        individuals[chosen_indices[i]].fitness.values[idx] - \
                        individuals[chosen_indices[j]].fitness.values[idx]
                    dist += val * val
                distances[i][j] = dist
                distances[j][i] = dist
            distances[i][i] = -1

        # Insert sort is faster than quick sort for short arrays
        for i in range(inds_len):
            for j in range(1, inds_len):
                idx = j
                while idx > 0 and distances[i][j] < distances[i][sorted_indices[i][idx - 1]]:
                    sorted_indices[i][idx] = sorted_indices[i][idx - 1]
                    idx -= 1
                sorted_indices[i][idx] = j

        size = inds_len
        to_remove = []
        while size > pop_size:
            # Search for minimal distance
            min_pos = 0
            for i in range(1, inds_len):
                for j in range(1, size):
                    dist_i_sorted_j = distances[i][sorted_indices[i][j]]
                    dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

                    if dist_i_sorted_j < dist_min_sorted_j:
                        min_pos = i
                        break
                    elif dist_i_sorted_j > dist_min_sorted_j:
                        break

            # Remove minimal distance from sorted_indices
            for i in range(inds_len):
                distances[i][min_pos] = float("inf")
                distances[min_pos][i] = float("inf")

                for j in range(1, size - 1):
                    if sorted_indices[i][j] == min_pos:
                        sorted_indices[i][j] = sorted_indices[i][j + 1]
                        sorted_indices[i][j + 1] = min_pos

            # Remove corresponding individual from chosen_indices
            to_remove.append(min_pos)
            size -= 1

        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]

    return [individuals[i] for i in chosen_indices]


# Auxiliary algorithmic functions for spea2_selection
# This code is a part of DEAP library (Library URL: https://github.com/DEAP/deap).
def _randomized_select(array: List[float], begin: int, end: int, i: float) -> float:
    """
    Allows to select the ith smallest element from array without sorting it.
    Runtime is expected to be O(n).
    """
    if begin == end:
        return array[begin]
    q = _randomized_partition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomized_select(array, begin, q, i)
    else:
        return _randomized_select(array, q + 1, end, i - k)


def _randomized_partition(array: List[float], begin: int, end: int) -> int:
    i = randint(begin, end)
    array[begin], array[i] = array[i], array[begin]
    return _partition(array, begin, end)


def _partition(array: List[float], begin: int, end: int) -> int:
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j
