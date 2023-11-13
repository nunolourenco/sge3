import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
import numpy as np

def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    if params['META_MUTATION']:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'mutation_probs': [params['PROB_MUTATION'] for _ in genotype]}
    else:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}

def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


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

    params['EXPERIMENT_NAME'] += "/" + "start_mut_prob_" + str(params['PROB_MUTATION']) + "/" + "prob_mut_probs_" + str(params['META_MUT_RATE']) +"/gauss_" + str(params['META_MUT_EFFECT'])

    logger.prepare_dumps()
    np.random.seed(int(params['SEED']))
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])

def meta_mutation(ind):
    mutation_probabilities = ind['mutation_probs']
    new_p = []
    for nt in mutation_probabilities:
        if np.random.uniform() < params['META_MUT_RATE']:
            gauss = np.random.normal(0.0,params['META_MUT_EFFECT'])
            nt = max(nt+gauss,0)
            nt = min(nt,1)
        new_p.append(nt)
    ind['mutation_probs'] = new_p
    return ind

def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    it = 0
    while it <= params['GENERATIONS']:
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)
        population.sort(key=lambda x: x['fitness'])

        logger.evolution_progress(it, population)
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if np.random.uniform() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            if params['META_MUTATION']:
                ni = meta_mutation(ni)
                ni = mutate_level(ni)
            else:
                ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)
        population = new_population
        it += 1

