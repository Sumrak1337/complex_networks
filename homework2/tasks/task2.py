import networkx as nx
import matplotlib.pyplot as plt
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
        pos = nx.spring_layout(sub_graph)
        nx.draw_networkx_edges(sub_graph, pos=pos)
        # TODO: nicer graphs
        # TODO: calculate modularity
        for i, nodes in enumerate(mm):
            nx.draw_networkx_nodes(sub_graph, pos=pos, nodelist=nodes, node_color=colors[i+1])
        print('modularity: ', nx.algorithms.community.modularity(sub_graph, mm))
        # plt.show()

        # Edge-betweenness
        eb_graph = sub_graph.copy()
        part_seq = []
        for _ in range(nx.number_of_edges(eb_graph)):
            eb = nx.edge_betweenness_centrality(eb_graph)
            mve = max(nx.edges(eb_graph), key=eb.get)
            eb_graph.remove_edge(*mve)
            partition = list(nx.connected_components(eb_graph))
            part_seq.append(partition)

        # modularity_seq = [nx.algorithms.community.modularity(sub_graph, part) for part in part_seq]  # need for graph
        best_partition = max(part_seq, key=lambda p: nx.algorithms.community.modularity(sub_graph, p))
        # TODO: also make nicer graph
        nx.draw_networkx_edges(sub_graph, pos=pos)
        for i, nodes in enumerate(best_partition):
            nx.draw_networkx_nodes(sub_graph, pos=pos, nodelist=nodes, node_color=colors[i+1])
        print('modularity: ', nx.algorithms.community.modularity(sub_graph, best_partition))
        # plt.show()

