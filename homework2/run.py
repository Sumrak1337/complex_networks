import networkx as nx
from pathlib import Path

from utils import get_logger
from homework2.tasks.task1 import Task1
# from homework2.tasks.task2 import Task2

log = get_logger(__name__)

CLEAR_DATA_ROOT = Path(__file__).parent.absolute() / 'clear_data'
RESULTS_ROOT = Path(__file__).parent.absolute() / 'results'


def main():
    file = 'vk_friends_graph.gexf'
    graph = nx.read_gexf(CLEAR_DATA_ROOT / file)
    for Task in [Task1,
                 # Task2,
                 ]:
        Task(graph, file).run()


if __name__ == '__main__':
    main()
