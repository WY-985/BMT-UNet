import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        #self.log = open(filename, 'a')
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as f:
            f.write(message)

    def flush(self):
        pass








