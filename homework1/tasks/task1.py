import networkx as nx
import pandas as pd

from utils import GraphData, get_logger
from homework1.task_defaults import DATA_ROOT

log = get_logger(__name__)


class Task1:
    prefix = 'task1'

    def __init__(self):
        self.graph = None

    def run(self):
        graphs = [
            'CA-AstroPh.txt',
            # 'vk.csv',
            # 'vk_friends_graph.gexf',
            # 'web-Google.txt'
        ]

        for graph_tag in graphs:
            log.info(f'Graph {graph_tag}:')
            graph = self.get_graph(graph_tag)

            n_nodes = len(nx.nodes(graph))
            n_edges = len(nx.edges(graph))
            max_edges = n_nodes * (n_nodes - 1) if graph_tag == GraphData.WEB_GRAPH.value else n_nodes * (n_nodes - 1)
            n_weak_cc = len(list(nx.weakly_connected_components(graph))) if graph_tag == GraphData.WEB_GRAPH.value else len(list(nx.connected_components(graph)))
            n_strong_cc = len(list(nx.strongly_connected_components(graph))) if graph_tag == GraphData.WEB_GRAPH.value else None
            weak_cc = max(nx.weakly_connected_components(graph), key=len) if graph_tag == GraphData.WEB_GRAPH.value else max(nx.connected_components(graph), key=len)
            strong_cc = max(nx.strongly_connected_components(graph), key=len) if graph_tag == GraphData.WEB_GRAPH.value else None

            log.info(f'# of nodes: {n_nodes}')
            log.info(f'# of edges: {n_edges}')
            log.info(f'Density: {n_edges / max_edges:.4f}')
            log.info(f'Max weakly connected components: {n_weak_cc}')
            log.info(f'Weakly percentile: {len(weak_cc) / n_nodes:.4f}')
            if graph_tag == GraphData.WEB_GRAPH.value:
                log.info(f'Max strongly connected components: {n_strong_cc}')
                log.info(f'Strongly percentile: {len(strong_cc) / n_nodes:.4f}')

    def get_graph(self, graph_tag):
        if graph_tag == GraphData.ASTROPH_GRAPH.value:
            graph = nx.Graph()
            with open(DATA_ROOT / graph_tag, 'r') as f:
                for line in f.readlines():
                    if '#' in line:
                        continue
                    source, target = line.split()
                    graph.add_edge(source, target)
            return graph

        if graph_tag == GraphData.VK_GRAPH.value:
            vk = pd.read_csv(DATA_ROOT / graph_tag).drop(['t', 'h'], axis=1)
            graph = nx.from_pandas_edgelist(vk, source='u', target='v')
            return graph

        if graph_tag == GraphData.VK_FRIENDS_GRAPH.value:
            graph = nx.read_gexf(DATA_ROOT / graph_tag)
            graph = self.to_undirected(graph)
            return graph

        if graph_tag == GraphData.WEB_GRAPH.value:
            graph = nx.DiGraph()
            with open(DATA_ROOT / graph_tag, 'r') as f:
                for line in f.readlines():
                    if '#' in line:
                        continue
                    source, target = line.split()
                    graph.add_edge(source, target)
            return graph

    @staticmethod
    def to_undirected(graph):
        undirected_graph = nx.Graph()
        for edge in nx.edges(graph):
            undirected_graph.add_edge(*edge)
        return undirected_graph
