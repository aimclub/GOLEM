from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationType
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.core.optimisers.opt_history_objects.individual import Individual


class ReproductionController:
    """
    Task of the Reproduction Controller is to reproduce population
    while keeping population size as specified in optimizer settings.

    Args:
        parameters: genetic algorithm parameters.
        selection: operator used in reproduction.
        mutation: operator used in reproduction.
        crossover: operator used in reproduction.
        window_size: size in iterations of the moving window to compute reproduction success rate.
    """

    def __init__(self,
                 parameters: GPAlgorithmParameters,
                 selection: Selection,
                 mutation: Mutation,
                 crossover: Crossover,
                 verifier: GraphVerifier):
        self.parameters = parameters
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover
        self._minimum_valid_ratio = parameters.required_valid_ratio * 0.5

        self._log = default_log(self)

    def reproduce(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
           Implements additional checks on population to ensure that population size
           follows required population size.
        """
        selected_individuals = self.selection(population, self.parameters.pop_size)
        new_population = self.crossover(selected_individuals)
        new_population = self._mutate_over_population(new_population, evaluator)
        return new_population

    def _mutate_over_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        target_pop_size = self.parameters.pop_size
        max_tries = target_pop_size * MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
        max_attempts_count = self.parameters.max_num_of_operator_attempts
        mutation_fun = partial(self._mutation_n_evaluation, evaluator=evaluator)
        population_descriptive_ids_mapping = {ind.graph.descriptive_id: ind for ind in population}

        # mutations counters
        mutation_types = self.parameters.mutation_types
        mutation_count = {mutation_type: 0 for mutation_type in mutation_types}
        mutation_count_for_each_ind = {descriptive_id: copy(mutation_count)
                                       for descriptive_id in population_descriptive_ids_mapping}
        mutation_tries_for_each_ind = {descriptive_id: copy(mutation_count)
                                       for descriptive_id in population_descriptive_ids_mapping}

        # prepare one mutation for each individual in population
        # mutation_type is None, let Mutation() choose
        mutation_queue = Queue()
        for descriptive_id, individual in zip(population_descriptive_ids_mapping, population):
            mutation_queue.put((descriptive_id, individual, None))

        # run infinite cycle with evaluation in parallel
        new_population = list()
        with Parallel(n_jobs=self.mutation.requirements.n_jobs, return_as='generator') as parallel:
            ind_generator = parallel(delayed(mutation_fun)(mutation_queue) for _ in [1] * 5) # cycle([1]))
            for try_num, (parent_descriptive_id, mutation_type, new_ind) in enumerate(ind_generator):
                mutation_tries_for_each_ind[parent_descriptive_id][mutation_type] += 1
                if new_ind:
                    descriptive_id = new_ind.graph.descriptive_id
                    if descriptive_id not in self._pop_graph_descriptive_ids:
                        new_population.append(new_ind)
                        self._pop_graph_descriptive_ids.add(descriptive_id)
                        if len(new_population) >= target_pop_size:
                            break

                        # count mutations
                        mutation_count_for_each_ind[parent_descriptive_id][mutation_type] += 1
                        mutation_count[mutation_type] += 1

                        # choose new mutation for individual, rely on check probabilities
                        mutation_probabilities = self.mutation._operator_agent.get_action_probs()
                        # potential place for error if order in
                        # mutation_probabilities and mutation_real_propabilities differ
                        # needs to fix with all mabs and random agent
                        mutation_real_probabilities = [mutation_count / self.parameters.pop_size
                                                       for mutation_count in mutation_count.values()]

                        allowed_mutation_types = [mutation_type
                                                  for mutation_type, prob, real_prob in
                                                  zip(mutation_types,
                                                      mutation_probabilities,
                                                      mutation_real_probabilities)
                                                  if prob > real_prob]

                        if not allowed_mutation_types:
                            raise ValueError(f"Sum of mutation_probabilities is not equal to 1."
                                             f"Check _operator_agent in mutation.")

                        # get the most rare mutation for all inds
                        stop = False
                        lowest_mutation_count = (None, None, 0)
                        for _graph_id, _graph_id_mutation_count in mutation_count_for_each_ind.items():
                            for _mutation_type, _mutation_count in _graph_id_mutation_count.items():
                                if _mutation_type in allowed_mutation_types:
                                    if _mutation_count == 0:
                                        lowest_mutation_count = (_graph_id, _mutation_type, _mutation_count)
                                        stop = True
                                    elif _mutation_count < lowest_mutation_count[-1]:
                                        lowest_mutation_count = (_graph_id, _mutation_type, _mutation_count)
                                if stop:
                                    break
                            if stop:
                                break
                        graph_id_to_mutate, mutation_type, _ = lowest_mutation_count
                        mutation_queue.put((graph_id_to_mutate,
                                            population_descriptive_ids_mapping[graph_id_to_mutate],
                                            mutation_type))
                if try_num >= max_tries:
                    break

        self._check_final_population(new_population)

        return new_population

    def _check_final_population(self, population: PopulationT) -> None:
        """ If population do not achieve required length return a warning or raise exception """
        target_pop_size = self.parameters.pop_size
        helpful_msg = ('Check objective, constraints and evo operators. '
                       'Possibly they return too few valid individuals.')

        if len(population) < target_pop_size * self._minimum_valid_ratio:
            raise EvaluationAttemptsError('Could not collect valid individuals'
                                          ' for population.' + helpful_msg)
        elif len(population) < target_pop_size:
            self._log.warning(f'Could not achieve required population size: '
                              f'have {len(population)},'
                              f' required {target_pop_size}!\n' + helpful_msg)

    def _mutation_n_evaluation(self, i, mutation_queue: Queue,
                               evaluator: EvaluationOperator):
        try:
            # wait timeout in seconds for new task to reduce probability of process flooding
            descriptive_id, individual, mutation_type = mutation_queue.get(timeout=1)
            individual, mutation_type, applied = self.mutation._mutation(individual, mutation_type=mutation_type)
            if applied and individual and self.verifier(individual.graph):
                individuals = evaluator([individual])
                if individuals:
                    return descriptive_id, mutation_type, individuals[0]
        except queue.Empty:
            pass
        return descriptive_id, mutation_type, None
