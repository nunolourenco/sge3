import json
import multiprocessing as mp
import sys
from queue import Empty
from types import ModuleType
import logging
logging.basicConfig(filename='python_log.txt', format='%(asctime)s:%(process)d:%(thread)d:%(message)s', level=logging.INFO)  # set to DEBUG for debug info ;)


class Worker(mp.Process):
    def __init__(self, consume, produce):
        super(Worker, self).__init__()
        self.consume = consume
        self.produce = produce
        self.stop = mp.Value('b', False)

    def run(self):
        # START LINUX: used to receive Memory Error faster in Linux
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2 ** 30, 2 ** 30))  # 2 ** 30 == 1GB in bytes
        except ImportError:
            pass
        # END LINUX:
        while True:
            exception = None
            self.stop.value = False
            script = self.consume.get()
            if script:
                help_globals = {'stop': self.stop}
                try:
                    exec(script, help_globals)
                except BaseException as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    exception = '{} {} {}'.format(exc_type, exc_obj, e.args)
                if exception:
                    self.produce.put({'exception': exception})
                else:
                    self.produce.put({key: value for key, value in help_globals.items()
                                      if not callable(value) and             # cannot be a function
                                      not isinstance(value, ModuleType) and  # cannot be a module
                                      key not in ['__builtins__', 'stop']})  # cannot be builtins or synchronized objects
                del help_globals
            else:
                break

    def stop_current(self):
        self.stop.value = True

if __name__ == '__main__':
    consume = mp.Queue()
    produce = mp.Queue()
    p = Worker(consume, produce)
    p.start()
    while True:
        try:
            message = input()
            logging.debug('Received input')
        except EOFError as err:
            # No data was read with input()
            # HeuristicLab is not running anymore
            # stop thread
            consume.put(None)
            if not p.join(5):
                p.terminate()
            break

        # do not trust user input
        try:
            message_dict = json.loads(message)
        except json.decoder.JSONDecodeError as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            exception = '{} {} {}'.format(exc_type, exc_obj, e.args)
            logging.debug(exception)
            logging.debug(message)
            print(json.dumps({'exception': exception}), flush=True)
            continue
        consume.put(message_dict['script'])
        try:
            results = produce.get(block=True, timeout=message_dict['timeout'])
        except Empty:
            results = None
        if not results:
            p.stop_current()
            try:
                produce.get(block=True, timeout=message_dict['timeout'] * 10)
            except Empty:
                # START: Used to terminate worker process if it does not return
                # Possible reasons: OS X does not throw a MemoryError and might kill the worker itself
                #                   worker just takes too long in general
                p.terminate()
                consume = mp.Queue()
                produce = mp.Queue()
                p = Worker(consume, produce)
                p.start()
                logging.debug('terminated worker')
                # END:
            print(json.dumps({'exception': 'Timeout occurred.'}), flush=True)
            logging.debug('Sent output timeout')
        elif 'exception' in results:
            print(json.dumps(results), flush=True)
            logging.debug('Sent output exception')
        else:
            ret_message_dict = {}
            for v in message_dict['variables']:
                if v in results:
                    ret_message_dict[v] = list(results[v]) if isinstance(results[v], set) else results[v]
            print(json.dumps(ret_message_dict), flush=True)
            logging.debug('Sent output normal')
