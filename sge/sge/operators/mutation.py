import copy
import random
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
            if random.random() < pmutation:
                current_value = p['genotype'][at_gene][position_to_mutate]
                choices = []
                if p['tree_depth'] >= grammar.get_max_depth():
                    possibilities = grammar.get_shortest_path()[nt][1:]
                    rule = random.choice(possibilities)
                    expansion_possibility = grammar.get_grammar()[nt].index(rule)
                    p['genotype'][at_gene][position_to_mutate] = expansion_possibility
                else:
                    choices = list(range(0, size_of_genes[nt]))
                    choices.remove(current_value)
                    p['genotype'][at_gene][position_to_mutate] = random.choice(choices)
    return p
