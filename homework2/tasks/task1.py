import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from homework2.task_defaults import RESULTS_ROOT

from utils import get_logger, AbstractTask

log = get_logger(__name__)

colors = list(mcolors.TABLEAU_COLORS)


class Task1(AbstractTask):
    prefix = 'task1'

    def __init__(self, graph: nx.Graph, fname: str):
        super().__init__(graph=graph, fname=fname)
        self.G = nx.to_undirected(self.G)

    def run(self):
        # Get the largest connectivity component
        lcc = max(nx.connected_components(self.G), key=len)
        sub_graph = self.G.subgraph(lcc)
        pos = nx.spring_layout(sub_graph)

        # Current subgraph
        self.plot_network(subgraph=sub_graph,
                          nodelist=nx.nodes(sub_graph),
                          pos=pos,
                          title='Original subgraph',
                          tag='original',
                          color=colors[0])

        # Get the maximum graph clique
        all_cliques = list(nx.find_cliques(sub_graph))
        max_clique = sorted(all_cliques, key=len, reverse=True)[0]
        self.plot_network(subgraph=sub_graph,
                          nodelist=max_clique,
                          pos=pos,
                          title='Graph with maximal clique',
                          tag='max_clique',
                          color=colors[1],
                          label=f'Max clique={len(max_clique)}')

        # Get k-core
        max_core_nodes = nx.k_core(sub_graph).nodes()
        self.plot_network(subgraph=sub_graph,
                          nodelist=max_core_nodes,
                          pos=pos,
                          title='Graph with maximal k-core nodes',
                          tag='core',
                          color=colors[2],
                          label=f'Max k-core={len(max_core_nodes)}')

    @staticmethod
    def plot_network(subgraph, nodelist, pos, title, tag, color, label=None):
        other_nodes = nx.nodes(subgraph) - nodelist

        plt.figure(figsize=(16, 9))
        plt.title(f'{title}')

        # Draw specific nodes
        nx.draw_networkx_nodes(subgraph,
                               pos=pos,
                               nodelist=nodelist,
                               node_color=color,
                               node_size=200,
                               label=label,
                               alpha=0.5)
        # Draw other nodes
        nx.draw_networkx_nodes(subgraph,
                               pos=pos,
                               nodelist=other_nodes,
                               node_color=colors[0],
                               node_size=200,
                               alpha=0.5)
        # Draw labels
        nx.draw_networkx_labels(subgraph,
                                pos=pos)
        # Draw edges
        nx.draw_networkx_edges(subgraph,
                               pos=pos,
                               alpha=0.3)
        plt.tight_layout()
        if label is not None:
            plt.legend()
        plt.savefig(RESULTS_ROOT / f'{tag}.png')
