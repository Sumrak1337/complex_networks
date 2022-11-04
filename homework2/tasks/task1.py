import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from homework2.task_defaults import RESULTS_ROOT

from utils import get_logger, AbstractTask

log = get_logger(__name__)

colors = list(mcolors.TABLEAU_COLORS)


class Task1(AbstractTask):
    prefix = 'task1'
    # TODO: rewrite plot creating with one method

    def __init__(self, graph: nx.Graph, fname: str):
        super().__init__(graph=graph, fname=fname)
        self.G = nx.to_undirected(self.G)

    def run(self):
        # Get the largest connectivity component
        lcc = max(nx.connected_components(self.G), key=len)
        sub_graph = self.G.subgraph(lcc)

        # Current subgraph
        pos = nx.spring_layout(sub_graph)
        plt.figure(figsize=(20, 10))
        plt.title('Original subgraph')
        nx.draw(sub_graph,
                pos=pos,
                node_size=100,
                node_color=colors[0],
                with_labels=True)
        plt.tight_layout()
        plt.savefig(RESULTS_ROOT / 'original.png')

        # Get the maximum graph clique
        all_cliques = list(nx.find_cliques(sub_graph))
        max_clique = sorted(all_cliques, key=len, reverse=True)[0]
        other_nodes = nx.nodes(sub_graph) - max_clique

        plt.figure(figsize=(20, 10))
        plt.title('Graph with maximal clique')
        nx.draw_networkx_nodes(sub_graph,
                               pos=pos,
                               nodelist=max_clique,
                               node_color=colors[1],
                               node_size=100,
                               label=f'Max clique={len(max_clique)}',
                               alpha=0.5)
        nx.draw_networkx_nodes(sub_graph,
                               pos=pos,
                               nodelist=other_nodes,
                               node_color=colors[0],
                               node_size=100,
                               alpha=0.5)
        nx.draw_networkx_labels(sub_graph,
                                pos=pos)
        nx.draw_networkx_edges(sub_graph, pos=pos, alpha=0.5)
        nx.draw(sub_graph, pos=pos, alpha=0.)
        plt.tight_layout()
        plt.legend()
        plt.savefig(RESULTS_ROOT / 'max_clique.png')

        # Get k-core
        max_core_nodes = nx.k_core(sub_graph).nodes()
        plt.figure(figsize=(20, 10))
        plt.title('Graph with maximal k-core nodes')
        nx.draw_networkx_nodes(sub_graph,
                               pos=pos,
                               nodelist=max_core_nodes,
                               node_color=colors[2],
                               node_size=100,
                               label=f'Max k-core={len(max_core_nodes)}',
                               alpha=0.5)
        nx.draw_networkx_nodes(sub_graph,
                               pos=pos,
                               nodelist=other_nodes,
                               node_color=colors[0],
                               node_size=100,
                               alpha=0.5)
        nx.draw_networkx_labels(sub_graph,
                                pos=pos)
        nx.draw_networkx_edges(sub_graph,
                               pos=pos,
                               alpha=0.5)
        nx.draw(sub_graph, pos=pos, alpha=0.)
        plt.tight_layout()
        plt.legend()
        plt.savefig(RESULTS_ROOT / 'core.png')
