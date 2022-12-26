import random
import math

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

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
            avg_degree = 2 * n_edges / n_nodes
            greater_avg_degree = np.sum(np.array([True if value > avg_degree
                                                  else False
                                                  for value in dict(graph.degree()).values()]))

            subgraph = nx.subgraph(graph, weak_cc)
            n = min(500, len(weak_cc))
            random.seed(42)
            random_nodes = random.sample(list(nx.nodes(subgraph)), n)
            all_paths = []
            for i in tqdm(range(n - 1)):
                for j in range(i + 1, n):
                    s_path = nx.shortest_path_length(subgraph,
                                                     source=random_nodes[i],
                                                     target=random_nodes[j])
                    all_paths.append(s_path)
            all_paths = pd.Series(all_paths)

            radius = np.inf
            diameter = -np.inf
            for node in tqdm(random_nodes):
                e = nx.eccentricity(subgraph, node)
                if e > diameter:
                    diameter = e
                if e < radius:
                    radius = e

            # Main features description for original graph
            log.info(f'# of nodes: {n_nodes}')
            log.info(f'# of edges: {n_edges}')
            log.info(f'Density: {n_edges / max_edges:.4f}')
            log.info(f'Max weakly connected components: {n_weak_cc}')
            log.info(f'Weakly percentile: {len(weak_cc) / n_nodes:.4f}')
            if graph_tag == GraphData.WEB_GRAPH.value:
                log.info(f'Max strongly connected components: {n_strong_cc}')
                log.info(f'Strongly percentile: {len(strong_cc) / n_nodes:.4f}')
            log.info(f"# of nodes' degree greater than average: {greater_avg_degree}")

            log.info('For the largest weakly connectivity component:')
            log.info(f'Radius: {radius}')
            log.info(f'Diameter: {diameter}')
            log.info(f'P90: {all_paths.quantile(q=0.9)}')

            # cluster coefficients
            local_cluster_coefs = []
            n_triangles = 0
            den = 0
            for node in nx.nodes(graph):
                nbs = list(nx.neighbors(graph, node))
                e = 0
                for i in range(len(nbs) - 1):
                    for j in range(i + 1, len(nbs)):
                        if graph.has_edge(nbs[i], nbs[j]):
                            n_triangles += 1
                            e += 1
                nv = len(nbs)
                degree = nx.degree(graph, node)
                den += 0 if degree < 2 else math.factorial(nv) / math.factorial(2) / math.factorial(nv - 2)
                local_cluster_coefs.append(0 if degree < 2 else 2 * e / degree / (degree - 1))

            log.info(f'# of triangles: {n_triangles // 3}')
            log.info(f'Average cluster coefficient: {np.sum(local_cluster_coefs) / n_nodes:.4f}')
            log.info(f'Global cluster coefficient: {n_triangles / den:.4f}')

            # degree distribution in simple and log-log

            # linear function

            # delete nodes

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
