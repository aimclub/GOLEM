
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
        mutation_count_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}
        all_mutations_count_for_each_ind = {descriptive_id: 0 for descriptive_id in population_descriptive_ids_mapping}
        mutation_tries_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}

        # prepare one mutation for each individual in population
        mutation_queue = Manager().Queue()
        for descriptive_id, individual in zip(population_descriptive_ids_mapping, population):
            # mutation_type is None, let Mutation() choose
            mutation_queue.put((descriptive_id, individual, None))

        # run cycle with evaluation in parallel
        # made with joblib.Parallel due to
        #     it is simple
        #     it is reliable (joblib/loky solves some problems with multiprocessing)
        #     joblib is in requirements
        new_population = list()
        with Parallel(n_jobs=self.mutation.requirements.n_jobs, return_as='generator') as parallel:
            ind_generator = parallel(delayed(mutation_fun)(mutation_queue) for _ in range(max_tries * 2))
            for try_num, (parent_descriptive_id, mutation_type, new_ind) in enumerate(ind_generator):
                if parent_descriptive_id is None:
                    continue
                mutation_tries_for_each_ind[parent_descriptive_id][mutation_type] += 1
                if new_ind:
                    mutation_count_for_each_ind[parent_descriptive_id][mutation_type] += 1
                    all_mutations_count_for_each_ind[parent_descriptive_id] += 1

                    descriptive_id = new_ind.graph.descriptive_id
                    if descriptive_id not in self._pop_graph_descriptive_ids:
                        # add ind to new population
                        new_population.append(new_ind)
                        self._pop_graph_descriptive_ids.add(descriptive_id)
                        if len(new_population) >= target_pop_size:
                            break

                        # filter mutations for individual, rely on probabilities
                        # place for error if mutation_types order in _operator_agent and in mutation_types is differ
                        allowed_mutation_types = []
                        mutation_probabilities = self.mutation._operator_agent.get_action_probs()
                        for mutation_type, mutation_probability in zip(mutation_types, mutation_probabilities):
                            real_prob = (mutation_count_for_each_ind[parent_descriptive_id][mutation_type] /
                                         all_mutations_count_for_each_ind[parent_descriptive_id])
                            if real_prob < mutation_probability:
                                allowed_mutation_types.append(mutation_type)

                        if allowed_mutation_types:
                            # choose next mutation with lowest tries count
                            next_mutation_type = min(allowed_mutation_types,
                                                     key=lambda mutation_type:
                                                     mutation_tries_for_each_ind[parent_descriptive_id][mutation_type])

                            mutation_queue.put((parent_descriptive_id,
                                                population_descriptive_ids_mapping[parent_descriptive_id],
                                                next_mutation_type))
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

    def _mutation_n_evaluation(self,
                               mutation_queue: Queue,
                               evaluator: EvaluationOperator):
        try:
            descriptive_id, individual, mutation_type = mutation_queue.get(timeout=0.1)
        except queue.Empty:
            # is there is no task, then return nothing
            return None, None, None

        individual, mutation_type, applied = self.mutation._mutation(individual, mutation_type=mutation_type)
        if applied and individual and self.verifier(individual.graph):
            individuals = evaluator([individual])
            if individuals:
                # if all is ok return all data
                return descriptive_id, mutation_type, individuals[0]
        # if something go wrong do not return new individual
        return descriptive_id, mutation_type, None
