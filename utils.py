import logging
import sys
from enum import Enum

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

from homework2.task_defaults import RESULTS_ROOT

colors = list(mcolors.TABLEAU_COLORS)


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

    @staticmethod
    def plot_modularity_networkx(subgraph, title, pos, nodelist, tag, labels=True, node_size=200):
        plt.figure(figsize=(16, 9))
        plt.title(f'{title} subgraph')
        nx.draw_networkx_edges(subgraph,
                               pos=pos,
                               alpha=0.3)
        for i, nodes in enumerate(nodelist):
            nx.draw_networkx_nodes(subgraph,
                                   pos=pos,
                                   nodelist=nodes,
                                   node_size=node_size,
                                   node_color=colors[(i + 2) % 10],
                                   alpha=0.5)
        if labels:
            nx.draw_networkx_labels(subgraph,
                                    pos=pos,
                                    font_size=14)
        plt.tight_layout()
        plt.savefig(RESULTS_ROOT / f'{tag}.png')
        plt.close('all')

    @staticmethod
    def plot_network(subgraph, nodelist, pos, title, tag, save_root, color=colors[0], label=None, labels=True, node_size=200):
        other_nodes = nx.nodes(subgraph) - nodelist

        plt.figure(figsize=(16, 9))
        plt.title(f'{title}')

        # Draw specific nodes
        nx.draw_networkx_nodes(subgraph,
                               pos=pos,
                               nodelist=nodelist,
                               node_color=color,
                               node_size=node_size,
                               label=label,
                               alpha=0.5)
        # Draw other nodes
        nx.draw_networkx_nodes(subgraph,
                               pos=pos,
                               nodelist=other_nodes,
                               node_color=colors[0],
                               node_size=node_size,
                               alpha=0.5)
        # Draw labels
        if labels:
            nx.draw_networkx_labels(subgraph,
                                    pos=pos,
                                    font_size=14)
        # Draw edges
        nx.draw_networkx_edges(subgraph,
                               pos=pos,
                               alpha=0.3)
        plt.tight_layout()
        if label is not None:
            plt.legend()
        plt.savefig(save_root / f'{tag}.png')

    @staticmethod
    def get_pdf(graph_tag, unique, heights, save_root, loglog=False):
        plt.figure(figsize=(16, 10))
        plt.xlabel('Degree')
        plt.ylabel('Density')

        if loglog:
            log_unique = np.array([np.log(val) if val > 0 else 1e-6 for val in unique])
            log_heights = np.array([np.log(val) if val > 0 else 1e-6 for val in heights])
            log_heights = log_heights / max(log_heights)
            fname = f'pdf_{graph_tag}_loglog.png'
            plt.title(f'PDF of {graph_tag} (LogLog)')
            plt.scatter(log_unique, log_heights)
        else:
            fname = f'pdf_{graph_tag}.png'
            plt.title(f'PDF of {graph_tag}')
            plt.bar(unique, heights / max(heights))

        plt.tight_layout()
        plt.savefig(save_root / fname)
        plt.close()

    @staticmethod
    def get_cdf(graph_tag, heights, save_root, loglog=False):
        plt.figure(figsize=(16, 10))
        plt.xlabel('Degree')
        plt.ylabel('Density')

        if loglog:
            fname = f'cdf_{graph_tag}_loglog.png'
            log_heights = [np.log(val) if val > 0 else 1e-6 for val in heights]
            plt.hist(log_heights, bins=len(heights), rwidth=0.85, density=True, cumulative=True)
            plt.title(f'CDF of {graph_tag} (LogLog)')
        else:
            fname = f'cdf_{graph_tag}.png'
            plt.hist(heights, bins=len(heights), rwidth=0.85, density=True, cumulative=True)
            plt.title(f'CDF of {graph_tag}')

        plt.tight_layout()
        plt.savefig(save_root / fname)
        plt.close()

    @staticmethod
    def get_linear_approximation(graph_tag, unique, heights, save_root):
        unique = unique.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(unique, heights)

        # y = a + b * x
        a = reg.intercept_
        b = reg.coef_[0]
        y = reg.predict(unique)

        plt.figure(figsize=(16, 10))
        plt.title("Linear Approximation with LSM")
        plt.scatter(unique, heights)
        plt.plot(unique, y, label=f'y={a:.2f} + {b:.2f}*x')
        plt.legend()
        plt.savefig(save_root / f'{graph_tag}_approximation.png')
        plt.close()


class GraphData(Enum):
    ASTROPH_GRAPH = 'CA-AstroPh.txt'
    VK_GRAPH = 'vk.csv'
    VK_FRIENDS_GRAPH = 'vk_friends_graph.gexf'
    WEB_GRAPH = 'web-Google.txt'
