import random

import networkx as nx
import pandas as pd
import numpy as np

from utils import AbstractTask, get_logger
from homework1.task_defaults import DATA_ROOT, RESULTS_ROOT

log = get_logger(__name__)


class Task2(AbstractTask):
    prefix = 'task2'

    def __init__(self):
        self.grqc = None
        self.gnp = None
        self.ws = None

    def run(self):
        self.preprocessing()

        for graph in [
            self.grqc,
            self.gnp,
            self.ws
        ]:
            log.info(f'{graph.name} graph')

            n_triangles = sum(nx.triangles(graph).values()) // 3
            log.info(f'# of triangles: {n_triangles}')
            log.info(f'Average Clustering Coefficient: {nx.average_clustering(graph):.4f}')
            log.info(f'Global Clustering Coefficient: {nx.transitivity(graph):.4f}')

            degree_seq = pd.Series([degree[1] for degree in nx.degree(graph)])
            unique, heights = np.unique(degree_seq, return_counts=True)

            log.info(f'Min degree: {np.min(degree_seq)}')
            log.info(f'Mean degree: {np.mean(degree_seq):.4f}')
            log.info(f'Max degree: {np.max(degree_seq)}')

            # pdf
            self.get_pdf(graph.name, unique, heights, RESULTS_ROOT)
            self.get_pdf(graph.name, unique, heights, RESULTS_ROOT, loglog=True)

            # cdf
            self.get_cdf(graph.name, heights, RESULTS_ROOT)
            self.get_cdf(graph.name, heights, RESULTS_ROOT, loglog=True)

        # for GR-QC
        degree_seq = pd.Series([degree[1] for degree in nx.degree(self.grqc)])
        unique, heights = np.unique(degree_seq, return_counts=True)
        self.get_linear_approximation(self.grqc.name, unique, heights, RESULTS_ROOT)

    def preprocessing(self):
        self.grqc = self.read_txt(DATA_ROOT / "CA-GrQc.txt")
        n = len(nx.nodes(self.grqc))
        m = len(nx.edges(self.grqc))
        all_edges = n * (n - 1) / 2

        self.gnp = nx.gnp_random_graph(n, m / all_edges, seed=42)
        self.gnp = nx.Graph(self.gnp, name='gnp')

        self.ws = nx.watts_strogatz_graph(n, 4, 0, seed=42)
        ws_nodes = list(nx.nodes(self.ws))
        random.seed(42)

        while len(nx.edges(self.ws)) < m:
            node1, node2 = random.choices(ws_nodes, k=2)
            self.ws.add_edge(node1, node2)
        self.ws = nx.Graph(self.ws, name='ws')

    @staticmethod
    def read_txt(path):
        graph = nx.Graph(name='CA-GrQc')
        with open(path, 'r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue
                source, target = line.split()
                graph.add_edge(source, target)
        return graph
