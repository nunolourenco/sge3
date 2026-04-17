import numpy as np
from sge.parameters import params
import json
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def calculate_unique_percentage(population_phenotypes):
    unique_phenotypes = set(population_phenotypes)
    unique_count = len(unique_phenotypes)
    total_count = len(population_phenotypes)
    unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0

    return unique_percentage

def evolution_progress(generation, pop):
    best = pop[0]
    fitness_samples = []
    test_error_samples = []
    depth_samples = []
    phenotypes = []
    
    for individual in pop:
        fitness_samples.append(individual['fitness'])
        test_error_samples.append(individual['other_info']['test_error'])
        depth_samples.append(individual['tree_depth'])
        phenotypes.append(individual['phenotype'])
    
    length_used_genotype_best = sum(best['mapping_values'])
    unique_percentage = calculate_unique_percentage(phenotypes)

    data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.2f' % (
        generation,
        best['fitness'],
        np.nanmean(fitness_samples),
        np.nanstd(fitness_samples),
        best.get('other_info', {}).get('test_error', np.nan),  # safe access
        np.nanmean(test_error_samples),
        np.nanstd(test_error_samples),
        best['tree_depth'],
        np.nanmean(depth_samples),
        np.nanmedian(depth_samples),
        length_used_genotype_best,
        unique_percentage
 )

    if params['VERBOSE']:
        print(data)

    save_progress_to_file(data)

    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    c = json.dumps(population)
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()