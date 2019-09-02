from sqlalchemy import Column, Integer, Float, JSON, Boolean, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EvolutionaryRun(Base):
    __tablename__ = 'evolutionary_run'
    id = Column(Integer, primary_key=True)
    run = Column(Integer)
    generation = Column(Integer)
    best_fitness = Column(Float)
    mean_fitness = Column(Float)
    std_fitness = Column(Float)

    def __repr__(self):
        return '%4d\t%.6e\t%.6e\t%.6e' % (self.generation, self.best_fitness, self.mean_fitness, self.std_fitness)


class PopulationSample(Base):
    __tablename__ = 'population_sample'
    id = Column(Integer, primary_key=True)
    run = Column(Integer)
    generation = Column(Integer)
    population = Column(JSON)


class Parameters(Base):
    __tablename__ = 'parameters'
    id = Column(Integer, primary_key=True)
    run = Column(Integer)
    popsize = Column(Integer)
    generations = Column(Integer)
    elitism = Column(Integer)
    seed = Column(Integer)
    prob_crossover = Column(Float)
    prob_mutation = Column(Float)
    tsize = Column(Integer)
    grammar = Column(String)
    experiment_name = Column(String)
    include_genotype = Column(Boolean)
    save_step = Column(Integer)
    verbose = Column(Float)
    min_tree_depth = Column(Integer)
    max_tree_depth = Column(Integer)
    parameters = Column(String)
