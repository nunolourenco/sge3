from os import path
import subprocess
import json
import sys



class ProgSys():
    """Fitness function for program synthesis problems. Grammars and datasets
    for 29 benchmark problems from doi.org/10.1145/2739480.2754769 are
    provided. Evaluation is done in a separate python process."""

    # constants required for formatting the code correctly
    INSERTCODE = "<insertCodeHere>"

    INDENTSPACE = "  "
    LOOPBREAK = "loopBreak"
    LOOPBREAKUNNUMBERED = "loopBreak%"
    LOOPBREAK_INITIALISE = "loopBreak% = 0"
    LOOPBREAK_IF = "if loopBreak% >"
    LOOPBREAK_INCREMENT = "loopBreak% += 1"
    FORCOUNTER = "forCounter"
    FORCOUNTERUNNUMBERED = "forCounter%"

    def __init__(self, dataset_train, dataset_test, grammar):
        # Initialise base fitness function class.
        super().__init__()
        
        self.training, self.test, self.embed_header, self.embed_footer = \
            self.get_data(dataset_train, dataset_test,
                          grammar)
        self.eval = self.create_eval_process()

    def evaluate(self, ind, **kwargs):
    
        dist = kwargs.get('dist', 'training')
        
        program = self.format_program(ind,
                                      self.embed_header, self.embed_footer)
        data = self.training if dist == "training" else self.test
        program = "{}\n{}\n".format(data, program)
        eval_json = json.dumps({'script': program, 'timeout': 1.0,
                                'variables': ['cases', 'caseQuality',
                                              'quality']})
        self.eval.stdin.write((eval_json+'\n').encode())
        self.eval.stdin.flush()
        result_json = self.eval.stdout.readline()

        result = json.loads(result_json.decode())

        if 'exception' in result and 'JSONDecodeError' in result['exception']:
            self.eval.stdin.close()
            self.eval = self.create_eval_process()

        if 'quality' not in result:
            result['quality'] = sys.maxsize
        return result['quality'], {}

    @staticmethod
    def create_eval_process():
        """create separate python process for evaluation"""
        return subprocess.Popen(['python3.7',
                                 'progsys/scripts/python_script_evaluation.py'],
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE)

    def format_program(self, individual, header, footer):
        """formats the program by formatting the individual and adding
        a header and footer"""
        last_new_line = header.rindex('\n')
        indent = header[last_new_line + len('\n'):len(header)]
        return header + self.format_individual(individual, indent) + footer

    def format_individual(self, code, additional_indent=""):
        """format individual by adding appropriate indentation and loop break
        statements"""
        code = code.replace("\le", "<=")
        code = code.replace("\ge", ">=")
        code = code.replace("\l", "<")
        code = code.replace("\g", ">")
        code = code.replace("\eb", "|")
        parts = code.split('\\n')
        indent = 0
        string_builder = ""
        for_counter_number = 0
        first = True
        for part in parts:
            line = part.strip()
            # remove indentation if bracket is at the beginning of the line
            while line.startswith(":}"):
                indent -= 1
                line = line[2:].strip()

            # add indent
            if not first:
                string_builder += additional_indent
            else:
                first = False

            for i in range(0, indent):
                string_builder += self.INDENTSPACE

            # add indentation
            while line.endswith("{:"):
                indent += 1
                line = line[:len(line) - 2].strip()
            # remove indentation if bracket is at the end of the line
            while line.endswith(":}"):
                indent -= 1
                line = line[:len(line) - 2].strip()

            if self.LOOPBREAKUNNUMBERED in line:
                if self.LOOPBREAK_INITIALISE in line:
                    line = ""  # remove line
                elif self.LOOPBREAK_IF in line:
                    line = line.replace(self.LOOPBREAKUNNUMBERED,
                                        self.LOOPBREAK)
                elif self.LOOPBREAK_INCREMENT in line:
                    line = line.replace(self.LOOPBREAKUNNUMBERED,
                                        self.LOOPBREAK)
                else:
                    raise Exception("Python 'while break' is malformed.")
            elif self.FORCOUNTERUNNUMBERED in line:
                line = line.replace(self.FORCOUNTERUNNUMBERED,
                                    self.FORCOUNTER + str(for_counter_number))
                for_counter_number += 1

            # add line to code
            string_builder += line
            string_builder += '\n'

        return string_builder

    def get_data(self, train, test, grammar):
        """ Return the training and test data for the current experiment.
        A new get_data method is required to load from a sub folder and to
        read the embed file"""
        train_set = train
        test_set = test

        embed_file = path.join("grammars", "progsys", (grammar[-10:-4] + "-Embed.txt"))
        with open(embed_file, 'r') as embed:
            embed_code = embed.read()
        insert = embed_code.index(self.INSERTCODE)
        embed_header, embed_footer = "", ""
        if insert > 0:
            embed_header = embed_code[:insert]
            embed_footer = embed_code[insert + len(self.INSERTCODE):]
        with open(train_set, 'r') as train_file,\
                open(test_set, 'r') as test_file:
            return train_file.read(), test_file.read(), \
                   embed_header, embed_footer


if __name__ == '__main__':
    import sge
    dataset_train = "resources/progsys/Median/Train.txt"
    dataset_test = "resources/progsys/Median/Test.txt"
    grammar = "grammars/progsys/Median.bnf"
    eval_func = ProgSys(dataset_train=dataset_train, dataset_test=dataset_test, grammar=grammar)
    sge.evolutionary_algorithm(evaluation_function=eval_func)
