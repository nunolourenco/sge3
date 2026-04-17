"""
#    This file was adapted from the DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
"""

MUX_SELECT_LINES = 3
MUX_IN_LINES = 2 ** MUX_SELECT_LINES
MUX_TOTAL_LINES = MUX_SELECT_LINES + MUX_IN_LINES

input_names = ['s0', 's1', 's2', 'i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']
inputs = [[0] * MUX_TOTAL_LINES for i in range(2 ** MUX_TOTAL_LINES)]
outputs = [None] * (2 ** MUX_TOTAL_LINES)

for i in range(2 ** MUX_TOTAL_LINES):
    value = i
    divisor = 2 ** MUX_TOTAL_LINES
    # Fill the input bits
    for j in range(MUX_TOTAL_LINES):
        divisor /= 2
        if value >= divisor:
            inputs[i][j] = 1
            value -= divisor
    # Determine the corresponding output
    # The Select most valuable bit is s2
    indexOutput = MUX_SELECT_LINES
    for j, k in enumerate(inputs[i][:MUX_SELECT_LINES]):
        indexOutput += k * 2 ** j
    outputs[i] = inputs[i][indexOutput]

    to_evaluate = [dict(zip(input_names, i)) for i in inputs]


class Multiplexer_11:
    def evaluate(self, individual):
        """
        SOLUTION = "(i0 and (not s2) and (not s1) and (not s0)) or (i1 and (not s2) and (not s1) and (s0)) or (i2 and (not s2) and (s1) and (not s0)) or (i3 and (not s2) and (s1) and (s0)) or (i4 and s2 and not(s1) and not(s0)) or (i5 and s2 and (not s1) and s0) or (i6 and s2 and s1 and (not s0)) or (i7 and s2 and s1 and s0)"
        """

        error = len(inputs)
        try:
            program = compile("res = " + individual, '<string>', 'exec')
        except(SyntaxError, MemoryError):
            return 1000, -1
        for i, variables in enumerate(to_evaluate):
            exec(program, variables)
            res = variables['res']
            if res == outputs[i]:
                error -= 1
        return (error, {'generation': 0, "evals": 1, "test_error": 0.0})


if __name__ == "__main__":
    import sge

    eval_func = Multiplexer_11()
    sge.evolutionary_algorithm(evaluation_function=eval_func, parameters_file="parameters/standard.yml")