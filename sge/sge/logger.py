import numpy as np
from sge.parameters import params
from sge.utilities.run_info_orm import Base, EvolutionaryRun, PopulationSample, Parameters
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


engine = create_engine('sqlite:///%s.db' % params['EXPERIMENT_NAME'])
DBSession = sessionmaker(bind=engine)
session = DBSession()


def evolution_progress(generation, pop):
    fitness_samples = [i['fitness'] for i in pop]
    new_generation = EvolutionaryRun(run=params['RUN'], generation=generation, best_fitness=np.min(fitness_samples),
                                     mean_fitness=np.mean(fitness_samples),std_fitness=np.std(fitness_samples))
    session.add(new_generation)
    session.cpythonommit()
    if params['VERBOSE']:
        print(new_generation)
    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)


def save_step(generation, population):
    pop_dump = PopulationSample(run=params['RUN'], generation=generation,population=population)
    session.add(pop_dump)
    session.commit()


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    new_parameters = Parameters(**params_lower)
    session.add(new_parameters)
    session.commit()


def prepare_dumps():
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    save_parameters()
