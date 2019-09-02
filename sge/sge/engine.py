import random
import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters
)


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
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


def setup():
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def evolutionary_algorithm(evaluation_function=None):
    setup()
    population = list(make_initial_population())
    it = 0
    while it <= params['GENERATIONS']:
        for i in population:
            if i['fitness'] is None:
                evaluate(i, evaluation_function)
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

