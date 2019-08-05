import re
import random
from utilities import ordered_set


class Grammar:
    """Class that represents a grammar. It works with the prefix notation."""
    NT = "NT"
    T = "T"
    NT_PATTERN = "(<.+?>)"
    RULE_SEPARATOR = "::="
    PRODUCTION_SEPARATOR = "|"

    def __init__(self, grammar_path=None, init_depth=6, max_depth=17):
        if grammar_path is None:
            raise Exception("Grammar file not detected!!")
        else:
            self.grammar_file = grammar_path
        self.grammar = {}
        self.productions_labels = {}
        self.non_terminals, self.terminals = set(), set()
        self.ordered_non_terminals = ordered_set.OrderedSet()
        self.start_rule = None
        self.max_depth = max_depth
        self.max_init_depth = init_depth
        self.read_grammar()

    def read_grammar(self):
        """
        Reads a Grammar in the BNF format and converts it to a python dictionary
        This method was adapted from PonyGE version 0.1.3 by Erik Hemberg and James McDermott
        """
        with open(self.grammar_file, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip() != "":
                    if line.find(self.PRODUCTION_SEPARATOR):
                        left_side, productions = line.split(self.RULE_SEPARATOR)
                        left_side = left_side.strip()
                        if not re.search(self.NT_PATTERN, left_side):
                            raise ValueError("Left side not a non-terminal!")
                        self.non_terminals.add(left_side)
                        self.ordered_non_terminals.add(left_side)
                        # assumes that the first rule in the file is the axiom
                        if self.start_rule is None:
                            self.start_rule = (left_side, self.NT)
                        temp_productions = []
                        for production in [production.strip() for production in productions.split(self.PRODUCTION_SEPARATOR)]:
                            temp_production = []
                            if not re.search(self.NT_PATTERN, production):
                                if production == "None":
                                    production = ""
                                self.terminals.add(production)
                                temp_production.append((production, self.T))
                            else:
                                for value in re.findall("<.+?>|[^<>]*", production):
                                    if value != "":
                                        if re.search(self.NT_PATTERN, value) is None:
                                            sym = (value, self.T)
                                            self.terminals.add(value)
                                        else:
                                            sym = (value, self.NT)
                                        temp_production.append(sym)
                            temp_productions.append(temp_production)
                        if left_side not in self.grammar:
                            self.grammar[left_side] = temp_productions

    def count_number_of_options_in_production(self):
        number_of_options_by_non_terminal = {}
        g = self.grammar
        for nt in self.ordered_non_terminals:
            number_of_options_by_non_terminal.setdefault(nt, len(g[nt]))
        return number_of_options_by_non_terminal

    def mapping(self, mapping_rules, positions_to_map, needs_python_filter=False):
        output = []
        max_depth = self.recursive_mapping(mapping_rules, positions_to_map, self.start_rule, 0, output)
        output = "".join(output)
        if needs_python_filter:
            output = self.python_filter(output)
        return output, max_depth

    def recursive_mapping(self, mapping_rules, positions_to_map, current_sym, current_depth, output):
        depths = [current_depth]
        if current_sym[1] == self.T:
            output.append(current_sym[0])
        else:
            current_sym_pos = self.ordered_non_terminals.index(current_sym[0])
            choices = self.grammar[current_sym[0]]
            size_of_gene = self.count_number_of_options_in_production()
            if positions_to_map[current_sym_pos] >= len(mapping_rules[current_sym_pos]):
                if current_depth > self.max_depth:
                    # print "True"
                    possibilities = []
                    for index, option in enumerate(self.grammar[current_sym[0]]):
                        for s in option:
                            if s[0] == current_sym[0]:
                                break
                        else:
                            possibilities.append(index)
                    expansion_possibility = random.choice(possibilities)
                else:
                    expansion_possibility = random.randint(0, size_of_gene[current_sym[0]] - 1)
                mapping_rules[current_sym_pos].append(expansion_possibility)
            current_production = mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]]
            positions_to_map[current_sym_pos] += 1
            next_to_expand = choices[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self.recursive_mapping(mapping_rules, positions_to_map, next_sym, current_depth + 1, output))
        return max(depths)

    @staticmethod
    def python_filter(txt):
        """ Create correct python syntax.
        We use {: and :} as special open and close brackets, because
        it's not possible to specify indentation correctly in a BNF
        grammar without this type of scheme."""
        txt = txt.replace("\le", "<=")
        txt = txt.replace("\ge", ">=")
        txt = txt.replace("\l", "<")
        txt = txt.replace("\g", ">")
        txt = txt.replace("\eb", "|")
        indent_level = 0
        tmp = txt[:]
        i = 0
        while i < len(tmp):
            tok = tmp[i:i+2]
            if tok == "{:":
                indent_level += 1
            elif tok == ":}":
                indent_level -= 1
            tabstr = "\n" + "  " * indent_level
            if tok == "{:" or tok == ":}" or tok == "\\n":
                tmp = tmp.replace(tok, tabstr, 1)
            i += 1
            # Strip superfluous blank lines.
            txt = "\n".join([line for line in tmp.split("\n") if line.strip() != ""])
        return txt

    def __str__(self):
        grammar = self.grammar
        text = ""
        for key in self.ordered_non_terminals:
            text += key + " ::= "
            for options in grammar[key]:
                for option in options:
                    text += option[0]
                if options != grammar[key][-1]:
                    text += " | "
            text += "\n"
        return text


if __name__ == "__main__":
    random.seed(42)
    g = Grammar("grammars/regression.txt", 9)
    genome = [[0], [0, 3, 3], [0], [], [1, 1]]
    mapping_numbers = [0] * len(genome)
    print(g.mapping(genome, mapping_numbers, needs_python_filter=True))

