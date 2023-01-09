import random

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import GraphData, get_logger, AbstractTask
from homework1.task_defaults import DATA_ROOT, RESULTS_ROOT

log = get_logger(__name__)


class Task1(AbstractTask):
    prefix = 'task1'

    def run(self):
        graphs = [
            'CA-AstroPh.txt',
            'vk.csv',
            'vk_friends_graph.gexf',
            'web-Google.txt'
        ]

        for graph_tag in graphs:
            log.info(f'Graph {graph_tag}:')
            graph = self.get_graph(graph_tag)
            n_nodes = len(nx.nodes(graph))
            n_edges = len(nx.edges(graph))
            max_edges = n_nodes * (n_nodes - 1) if graph_tag == GraphData.WEB_GRAPH.value else n_nodes * (n_nodes - 1) / 2
            n_weak_cc = len(list(nx.weakly_connected_components(graph))) if graph_tag == GraphData.WEB_GRAPH.value else len(list(nx.connected_components(graph)))
            n_strong_cc = len(list(nx.strongly_connected_components(graph))) if graph_tag == GraphData.WEB_GRAPH.value else None
            weak_cc = max(nx.weakly_connected_components(graph), key=len) if graph_tag == GraphData.WEB_GRAPH.value else max(nx.connected_components(graph), key=len)
            strong_cc = max(nx.strongly_connected_components(graph), key=len) if graph_tag == GraphData.WEB_GRAPH.value else None
            avg_degree = 2 * n_edges / n_nodes
            greater_avg_degree = np.sum(np.array([True if value > avg_degree
                                                  else False
                                                  for value in dict(graph.degree()).values()]))

            n = min(100, len(strong_cc)) if graph_tag == GraphData.WEB_GRAPH.value else min(500, len(weak_cc))
            subgraph = nx.subgraph(graph, strong_cc) if graph_tag == GraphData.WEB_GRAPH.value else nx.subgraph(graph, weak_cc)
            random.seed(42)
            random_nodes = random.sample(list(nx.nodes(subgraph)), n)
            all_paths = []
            for i in tqdm(range(n - 1)):
                for j in range(i + 1, n):
                    s = random_nodes[i]
                    t = random_nodes[j]
                    s_path = nx.shortest_path_length(subgraph,
                                                     source=s,
                                                     target=t)
                    all_paths.append(s_path)
            all_paths = pd.Series(all_paths)
            ecc = list(nx.eccentricity(subgraph, v=random_nodes).values())

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
            log.info(f'Radius: {min(ecc)}')
            log.info(f'Diameter: {max(ecc)}')
            log.info(f'P90: {all_paths.quantile(q=0.9)}')

            # cluster coefficients
            graph = self.to_undirected(graph)
            n_triangles = sum(nx.triangles(graph).values()) // 3
            log.info(f'# of triangles: {n_triangles}')
            log.info(f'Average Clustering Coefficient: {nx.average_clustering(graph):.4f}')
            log.info(f'Global Clustering Coefficient: {nx.transitivity(graph):.4f}')

            # degree distribution in simple and log-log
            self.get_hist(graph, graph_tag)
            self.get_hist(graph, graph_tag, loglog=True)

            degree_seq = pd.Series([degree[1] for degree in nx.degree(graph)])
            unique, heights = np.unique(degree_seq, return_counts=True)

            log.info(f'Min degree: {np.min(degree_seq)}')
            log.info(f'Max degree: {np.max(degree_seq)}')
            log.info(f'Average degree: {np.mean(degree_seq):.4f}')

            # pdf
            self.get_pdf(graph_tag, unique, heights, RESULTS_ROOT)
            self.get_pdf(graph_tag, unique, heights, RESULTS_ROOT, loglog=True)

            # cdf
            self.get_cdf(graph_tag, heights, RESULTS_ROOT)
            self.get_cdf(graph_tag, heights, RESULTS_ROOT, loglog=True)

            # linear function
            self.get_linear_approximation(graph_tag, unique, heights, RESULTS_ROOT)

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
    def get_hist(graph, graph_tag, loglog=False):
        degree_seq = pd.Series([degree[1] for degree in nx.degree(graph)])
        unique, heights = np.unique(degree_seq, return_counts=True)

        if loglog:
            log_unique = [val if val > 0 else 1e-6 for val in unique]
            log_heights = [val if val > 0 else 1e-6 for val in heights]
            plt.figure()
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Density Degree Distribution (LogLog)")
            plt.scatter(log_unique, log_heights)
            plt.tight_layout()
            plt.savefig(RESULTS_ROOT / f'{graph_tag}_distr_loglog.png')
        else:
            plt.figure()
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Density Degree Distribution")
            plt.bar(unique, heights / sum(heights))
            plt.tight_layout()
            plt.savefig(RESULTS_ROOT / f'{graph_tag}_distr.png')

    @staticmethod
    def to_undirected(graph):
        undirected_graph = nx.Graph()
        for edge in nx.edges(graph):
            undirected_graph.add_edge(*edge)
        return undirected_graph
