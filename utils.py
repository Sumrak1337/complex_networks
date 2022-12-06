import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
import sys

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
    def plot_network(subgraph, nodelist, pos, title, tag, color=colors[0], label=None, labels=True, node_size=200):
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
        plt.savefig(RESULTS_ROOT / f'{tag}.png')
