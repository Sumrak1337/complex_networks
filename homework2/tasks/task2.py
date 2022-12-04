import networkx as nx
import matplotlib.colors as mcolors

from homework2.task_defaults import CLEAR_DATA_ROOT
from utils import get_logger, AbstractTask

colors = list(mcolors.TABLEAU_COLORS)
log = get_logger(__name__)


class Task2(AbstractTask):
    prefix = 'task2'

    def __init__(self,):
        self.graph = None

    def run(self):
        self.graph = nx.read_gexf(CLEAR_DATA_ROOT / 'vk_friends_graph.gexf')
        self.graph = nx.to_undirected(self.graph)

        # Get the largest connectivity component
        lcc = max(nx.connected_components(self.graph), key=len)
        sub_graph = self.graph.subgraph(lcc)

        # Modularity maximization
        mm = nx.algorithms.community.greedy_modularity_communities(sub_graph)
        pos = nx.spring_layout(sub_graph,
                               iterations=500,
                               seed=42)

        self.plot_modularity_networkx(subgraph=sub_graph,
                                      title='Modularity maximization',
                                      pos=pos,
                                      nodelist=mm,
                                      tag='max_modul')
        log.info(f'Modularity from Modularity Maximization: {nx.algorithms.community.modularity(sub_graph, mm):.4f}')

        # Edge-betweenness
        eb_graph = sub_graph.copy()
        part_seq = []
        for _ in range(nx.number_of_edges(eb_graph)):
            eb = nx.edge_betweenness_centrality(eb_graph)
            mve = max(nx.edges(eb_graph), key=eb.get)
            eb_graph.remove_edge(*mve)
            partition = list(nx.connected_components(eb_graph))
            part_seq.append(partition)

        best_partition = max(part_seq, key=lambda x: nx.algorithms.community.modularity(sub_graph, x))

        self.plot_modularity_networkx(subgraph=sub_graph,
                                      title='Edge-betweenness',
                                      pos=pos,
                                      nodelist=best_partition,
                                      tag='edge'
                                      )
        log.info(f'Modularity from Edge-Betweenness: {nx.algorithms.community.modularity(sub_graph, best_partition):.4f}')
