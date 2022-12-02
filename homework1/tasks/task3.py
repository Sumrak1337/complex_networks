import networkx as nx

from utils import AbstractTask
from homework1.preprocessing_utils import CLEAR_DATA_ROOT


class Task3(AbstractTask):
    prefix = 'task3'

    def __init__(self):
        self.G = nx.read_gexf(CLEAR_DATA_ROOT / 'vk_friends_graph.gexf')
        self.G = nx.to_undirected(self.G)

    def run(self):
        lcc = max(nx.connected_components(self.G), key=len)
        subgraph = nx.subgraph(self.G, lcc)

        print(subgraph)
