import random
import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def map_individual(ind):
    """
    Maps the genotype to phenotype and stores auxiliary info.
    """
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    ind['phenotype'] = phen
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    it = 0
    while it <= params['GENERATIONS']:
        # 1. Identify individuals that need evaluation
        to_evaluate = [ind for ind in population if ind['fitness'] is None]
        # 2. Map Genotypes to Phenotypes for all of them
        for ind in to_evaluate:
            map_individual(ind)
        #Batch vs Single
        if to_evaluate:
            phenotypes = [ind['phenotype'] for ind in to_evaluate]
            try:
                # --- ATTEMPT BATCH EVALUATION ---
                # Try passing the whole list of phenotypes to the evaluator
                results = evaluation_function.evaluate(phenotypes)
                
                # Check if the result is valid (must be a list of same length as input)
                if isinstance(results, (list, tuple)) and len(results) == len(to_evaluate):
                    # Assign results from batch
                    for ind, res in zip(to_evaluate, results):
                        ind['fitness'] = res[0]
                        ind['other_info'] = res[1]
                else:
                    # If result isn't a list or wrong length, force fallback
                    raise ValueError("Batch evaluation returned invalid format.")

            except (AttributeError, TypeError, ValueError, Exception):
                # --- FALLBACK TO SINGLE EVALUATION ---
                # If batch fails (e.g. function expects single item), loop manually
                for ind in tqdm(to_evaluate):
                    quality, other_info = evaluation_function.evaluate(ind['phenotype'])
                    ind['fitness'] = quality
                    ind['other_info'] = other_info
        population.sort(key=lambda x: x['fitness'])
        logger.evolution_progress(it, population)
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)
        population = new_population
        it += 1

