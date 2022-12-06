import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import get_logger, AbstractTask
from homework1.preprocessing_utils import RESULTS_ROOT


log = get_logger(__name__)

#TODO: ALL WRONG
class Task1(AbstractTask):
    prefix = 'task1'

    def __init__(self, graph: nx.Graph, fname: str):
        super().__init__(graph=graph, fname=fname)

        self.n_nodes = None
        self.n_edges = None
        self.density = None
        self.weak_connect = None
        self.percent_max_weak_connect = None
        self.strong_connect = None
        self.percent_max_strong_connect = None
        self.n_nodes_gr_avg = None
        self.radius = None
        self.diameter = None
        self.dist90 = None
        self.n_triangles = None
        self.avg_cluster_coef = None
        self.global_cluster_coef = None
        self.min_degree = None
        self.max_degree = None
        self.avg_degree = None

    def run(self):
        tr = nx.triangles(self.G)
        print(tr)
        return
        self.n_nodes = self.G.number_of_nodes()
        self.n_edges = self.G.number_of_edges()
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1)) / 2

        if self.fname == 'web-Google':
            self.weak_connect = nx.number_weakly_connected_components(self.G)
            large_weak_cc = max(nx.weakly_connected_components(self.G), key=len)
            large_weak_cc_g = self.G.subgraph(large_weak_cc).copy()
            self.percent_max_weak_connect = large_weak_cc_g.number_of_nodes() / self.n_nodes

            self.strong_connect = nx.number_strongly_connected_components(self.G)
            large_strong_cc = max(nx.strongly_connected_components(self.G), key=len)
            large_strong_cc_g = self.G.subgraph(large_strong_cc).copy()
            self.percent_max_strong_connect = large_strong_cc_g.number_of_nodes() / self.n_nodes
        else:
            self.weak_connect = nx.number_connected_components(self.G)
            large_cc = max(nx.connected_components(self.G), key=len)
            large_weak_cc_g = self.G.subgraph(large_cc).copy()
            self.percent_max_weak_connect = large_weak_cc_g.number_of_nodes() / self.n_nodes

        self.avg_degree = 2 * self.n_edges / self.n_nodes
        self.n_nodes_gr_avg = np.sum(np.array([True if value > self.avg_degree
                                               else False
                                               for value in dict(self.G.degree()).values()]))

        random.seed(42)
        random_node = random.choice(list(large_weak_cc_g.nodes()))
        subgraph_nodes = [random_node]
        new_subgraph_nodes = []
        n = large_weak_cc_g.number_of_nodes()
        n_nodes = 500 if n > 500 else n
        i = 0
        while len(new_subgraph_nodes) < n_nodes:
            nbs = list(large_weak_cc_g.neighbors(subgraph_nodes[i]))
            for nb in nbs:
                if nb not in subgraph_nodes:
                    subgraph_nodes.append(nb)
            new_subgraph_nodes.append(subgraph_nodes[i])
            i += 1

        subgraph = nx.subgraph(large_weak_cc_g, new_subgraph_nodes)

        self.radius = nx.radius(subgraph)
        self.diameter = nx.diameter(subgraph)

        if self.G.is_directed():
            self.G = self.G.to_undirected()

        adj_matrix = nx.to_numpy_array(self.G)
        self.n_triangles = int(np.trace(adj_matrix @ adj_matrix @ adj_matrix) / 6)

        local_cluster_coef = []
        n_neighbors = []
        # numerators = []
        # denominators = []
        for node in self.G.nodes():
            # nbs, lcc, num, den = self._node_calculating(node)
            nbs, lcc = self._node_calculating(node)
            n_neighbors.append(nbs)
            local_cluster_coef.append(lcc)
            # numerators.append(num)
            # denominators.append(den)

        self.avg_cluster_coef = sum(local_cluster_coef) / self.n_nodes
        # self.global_cluster_coef = sum(numerators) / sum(denominators)

        degree_seq = pd.Series([degree[1] for degree in nx.degree(self.G)])
        self.min_degree = min(degree_seq)
        self.max_degree = max(degree_seq)
        self._plot_pdf(degree_seq)
        self._plot_pdf(degree_seq, log_log=True)

        self._report()

    def _report(self):
        keys = ['# of nodes',
                '# of edges',
                'Density',
                '# of weakly connectivity components',
                'Percentage of max weakly connectivity component',
                '# of strong connectivity components',
                'Percentage of max strong connectivity component',
                '# nodes greater than average degree',
                'Radius of the largest weak connectivity component',
                'Diameter of the largest weak connectivity component',
                '90th percentile',
                '# of triangles',
                'Average cluster coefficient',
                'Global cluster coefficient',
                'Min degree',
                'Max degree',
                'Average degree'
                ]
        for key, value in zip(keys, self.get_values()):
            if value is None:
                continue
            if isinstance(value, float):
                print(f'{key:55}: {value:.5f}')
            else:
                print(f'{key:55}: {value}')

    @staticmethod
    def _c(n: int, k: int):
        return math.factorial(n) / math.factorial(k) / math.factorial(n - k)

    def _node_calculating(self, node):
        degree = nx.degree(self.G, node)
        if degree < 2:
            # return 0, 0, 0, 1
            return 0, 0
        nbs = len(list(nx.neighbors(self.G, node)))
        ego_graph = nx.ego_graph(self.G, node, radius=1)
        e = ego_graph.number_of_nodes() - degree
        lcc = 2 * e / degree / (degree - 1)
        # c = self._c(nbs, 2)
        # num = c * lcc
        # return nbs, lcc, num, c
        return nbs, lcc

    def _plot_pdf(self, degree_seq, log_log=False):
        postfix = ''
        plt.figure()
        plt.xlabel("Degree")
        plt.ylabel("Frequency")

        if log_log:
            plt.title('Density Degree Distribution (log_log)')
            degree_seq.plot(style='o', logx=True, logy=True)
            x = np.arange(len(degree_seq)).reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(x, degree_seq)
            y = reg.coef_[0] * x + reg.intercept_
            plt.plot(x, y)
            postfix = '_log_log'
        else:
            plt.title("Density Degree Distribution")
            unique, heights = np.unique(degree_seq, return_counts=True)
            plt.bar(unique, heights / sum(heights))
        plt.tight_layout()
        plt.savefig(RESULTS_ROOT / f'{self.fname}_degree_distr{postfix}.png')

