import logging
import sys


def get_logger(name) -> logging.Logger:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.DEBUG)

    if not log.hasHandlers():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)

    return log


class AbstractTask:
    def __init__(self, graph: nx.Graph, fname: str):
        self.G = graph
        self.fname = fname

    def run(self):
        raise NotImplementedError

    def get_features(self):
        features = []
        for k, v in self.__dict__.items():
            if k != 'G' and k != 'fname':
                features.append(k)
        return features

    def get_values(self):
        return [getattr(self, name) for name in self.get_features()]
