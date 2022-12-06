import networkx as nx
import matplotlib.colors as mcolors

from homework2.task_defaults import CLEAR_DATA_ROOT, RESULTS_ROOT
from utils import get_logger, AbstractTask

colors = list(mcolors.TABLEAU_COLORS)
log = get_logger(__name__)


class Task1(AbstractTask):
    prefix = 'task1'

    def __init__(self):
        self.graph = None

    def run(self):
        self.graph = nx.read_gexf(CLEAR_DATA_ROOT / 'vk_friends_graph.gexf')
        self.graph = nx.to_undirected(self.graph)

        self.plot_network(subgraph=self.graph,
                          nodelist=nx.nodes(self.graph),
                          pos=nx.spring_layout(self.graph,
                                               iterations=14,
                                               seed=42),
                          title='Original full graph',
                          tag='original_full',
                          save_root=RESULTS_ROOT,
                          color=colors[0],
                          labels=False)

        # Get the largest connectivity component
        lcc = max(nx.connected_components(self.graph), key=len)
        sub_graph = self.graph.subgraph(lcc)
        pos = nx.spring_layout(sub_graph,
                               iterations=500,
                               seed=42)

        # Current subgraph
        self.plot_network(subgraph=sub_graph,
                          nodelist=nx.nodes(sub_graph),
                          pos=pos,
                          title='Original subgraph',
                          tag='original',
                          save_root=RESULTS_ROOT,
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
                          save_root=RESULTS_ROOT,
                          label=f'Max clique={len(max_clique)}')

        # Get k-core
        max_core_nodes = nx.k_core(sub_graph).nodes()
        self.plot_network(subgraph=sub_graph,
                          nodelist=max_core_nodes,
                          pos=pos,
                          title='Graph with maximal k-core nodes',
                          tag='core',
                          color=colors[2],
                          save_root=RESULTS_ROOT,
                          label=f'Max k-core={len(max_core_nodes)}')
