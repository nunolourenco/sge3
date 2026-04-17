import copy
import numpy as np
import sge.grammar as grammar


def mutate(p, pmutation):
    p = copy.deepcopy(p)
    p['fitness'] = None
    size_of_genes = grammar.count_number_of_options_in_production()
    mutable_genes = [index for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar.get_non_terminals())[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if np.random.uniform() < pmutation:
                current_value = p['genotype'][at_gene][position_to_mutate]
                current_depth = current_value[1]
                if current_depth >= (grammar.get_max_depth() - grammar.get_shortest_path()[(nt,'NT')][0]):
                    choices = grammar.get_shortest_path()[(nt,'NT')][1:]
                    current_sym = grammar.get_grammar()[nt][current_value[0]]
                    if current_sym in choices and len(choices) > 1:
                        choices.remove(current_sym)
                    rule = choices[np.random.randint(0, len(choices))]
                    expansion_possibility = grammar.get_grammar()[nt].index(rule)
                    p['genotype'][at_gene][position_to_mutate] = [expansion_possibility, current_depth]

                else:
                    choices = list(range(0, size_of_genes[nt]))
                    choices.remove(current_value[0])
                    p['genotype'][at_gene][position_to_mutate] = [np.random.choice(choices), current_depth]
    return p
