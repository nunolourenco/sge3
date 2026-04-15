# sge3: And Implementation of Dynamic Structured Grammatical Evolution in Python 3

Structured Grammatical Evolution (SGE) is a recent Grammatical Evolution (GE) variant that aims at addressing some of its locality and redundancy issues. The SGE distinctive feature is having a one-to-one correspondence between genes and non-terminals of the grammar being used. If you use this code, a reference to the following work would be greatly appreciated:

```
@article{Lourenco2016,
 title={Unveiling the properties of structured grammatical evolution},
  author={Louren{\c{c}}o, Nuno and Pereira, Francisco B and Costa, Ernesto},
  journal={Genetic Programming and Evolvable Machines},
  volume={17},
  number={3},
  pages={251--289},
  year={2016},
  publisher={Springer}
}

@incollection{lourencco2018structured,
  title={Structured grammatical evolution: a dynamic approach},
  author={Louren{\c{c}}o, Nuno and Assun{\c{c}}{\~a}o, Filipe and Pereira, Francisco B and Costa, Ernesto and Machado, Penousal},
  booktitle={Handbook of Grammatical Evolution},
  pages={137--161},
  year={2018},
  publisher={Springer}
}
```

This project corresponds to a new implementation of the SGE engine. SGE has been criticised for the fact that we need to specify the maximum levels of recursion in order to remove it from the grammar beforehand. In this new version we specify the maximum tree depth (similarly to what happens in standard tree-based GP), and the algorithm adds the mapping numbers as required during the evolutionary search. Thus, we do not need to pre-process the grammar to remove the recursive productions. Additionally, we provide mechanisms and operators to ensure that the generated trees are always within the allowed limits.


As in for the SGE framework we provide the implementations of some problems that we used to test the DSGE. Extending it to your own needs should be fairly easy. 


When running the framework a folder called *dumps* will be created together with an additional one that corresponds to the experience. Inside, there will be directories for each run. Each run folder contains snapshots of the population at a given generation, and a file called *progress_report.csv*,  which is updated during the evolutionary run. By default we take snapshots of the population every iteration (SAVESTEP parameter in the configuration file). This can be changed, together with all the numeric values in the *configs* folder.

### Requirements
This code requires Python 3.11 or newer. The following dependencies are needed:
- numpy
- pandas
- torch
- pyyaml
- tqdm

### Installation

The project uses a `pyproject.toml` file for dependency management and configuration.

#### Using Poetry
1. Install Poetry if you haven't already: `curl -sSL https://install.python-poetry.org | python3 -`
2. Navigate to the `sge` directory: `cd sge`
3. Install dependencies: `poetry install`

#### Using venv
1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate` (on Linux/Mac) or `venv\Scripts\activate` (on Windows)
3. Navigate to the `sge` directory: `cd sge`
4. Install the package in editable mode: `pip install -e .`

### Execution

To run the algorithm you need a **grammar** and a **fitness function**.
The folder `examples/` contains the code for some benchmark problems used in Genetic Programming, and the folder ``grammars/`` contain the respective grammars. To run, for example, a Symbolic Regression problem, you can use the following command:

```python -m examples.symreg --experiment_name dumps/example --seed 791021 --parameters parameters/standard.yml```


#### Parameters

The folder `parameters/` contains an example of standard parameters to run. You can define the parameters on a file and specify them when executing the code. For example:

```
python3 -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/standard.yml
```

You can also add manually more parameters when calling the code without changing the parameter file. Here is an example where we define the seed:

```
python3 -m examples.symreg --grammar grammars/regression.pybnf --parameters parameters/standard.yml --seed 123
```

If you need to know the possible parameters, you can use the flag ``--help``. For example:

```
python -m examples.symreg --help
```

Here is the list of possible parameters, and how to call them.

| argument | type | description |
| --------------- | ----------- | ------------ |
| --parameters | str | Specifies the parameters file to be used. Must include the full file extension. | 
| --popsize | int | Specifies the population size. |
| --generations | int | Specifies the total number of generations.
| --elitism | int | Specifies the total number of individuals that should survive in each generation. |
| --prob_crossover | float | Specifies the probability of crossover usage. Float required. |
| --prob_mutation | float | Specifies the probability of mutation usage. Float required. |
| --tsize | int | Specifies the tournament size for parent selection. |
| --min_tree_depth | int | Specifies the initialisation tree depth. |
| --max_tree_depth | int | Specifies the maximum tree depth. |
| --grammar | str | specifies the path to the grammar file. |
| --experiment_name | str | Specifies the name of the folder where stats are going to be stored. |
| --run | int | Specifies the run number. |
| --seed | float | Specifies the seed to be used by the random number generator. |
| --include_genotype | bool | Specifies if the genotype is to be included in the log files |
| --save_step | int | Specifies how often stats are saved. |
| --verbose | bool | Turns on the verbose output of the program. |


### Support

Any questions, comments or suggestion should be directed to Nuno Lourenço ([naml@dei.uc.pt](mailto:naml@dei.uc.pt))

### Acknowledgments

I am grateful to my advisors Francisco B. Pereira and Ernesto Costa for their guidance during my PhD. I am also grateful to Filipe Assunção and Joaquim Ferrer for their help and comments on the development of this framework. 
