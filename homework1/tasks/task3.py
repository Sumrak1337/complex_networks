import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib import cm


from utils import AbstractTask, get_logger
from homework1.task_defaults import CLEAR_DATA_ROOT, RESULTS_ROOT

colors = list(mcolors.TABLEAU_COLORS)
log = get_logger(__name__)


class Task3(AbstractTask):
    prefix = 'task3'

    def __init__(self):
        self.G = nx.read_gexf(CLEAR_DATA_ROOT / 'vk_friends_graph.gexf')
        self.G = nx.to_undirected(self.G)

    def run(self):
        lcc = max(nx.connected_components(self.G), key=len)
        subgraph = nx.subgraph(self.G, lcc)

        pos = nx.spring_layout(subgraph,
                               seed=42)
        cmap = plt.get_cmap('jet')

        for centrality, c_tag in zip([nx.degree_centrality(subgraph),
                                      nx.closeness_centrality(subgraph),
                                      nx.betweenness_centrality(subgraph),
                                      nx.eigenvector_centrality(subgraph)],
                                     ['degree_cen',
                                      'closeness_cen',
                                      'betweenness_cen',
                                      'eigenvector_cen']):
            log.info(f'{c_tag}')
            first10 = sorted(centrality.items(), key=lambda item: item[1], reverse=True)[:10]
            log.info('The first 10 persons with the highest value:')
            for k, v in first10:
                print(f'{k:20} {v:.3f}')
            print()
            node_size = [v * 2e3 for v in centrality.values()]
            norm = mcolors.Normalize(vmin=min(centrality.values()), vmax=max(centrality.values()))
            scmp = cm.ScalarMappable(norm=norm, cmap=cmap)
            color = [scmp.to_rgba(v, alpha=0.5) for v in centrality.values()]
            self.plot_network(subgraph=subgraph,
                              nodelist=nx.nodes(subgraph),
                              pos=pos,
                              color=color,
                              title='default title',  # TODO: change
                              tag=c_tag,
                              save_root=RESULTS_ROOT,
                              node_size=node_size)

        """
        TODO:
        1. decay centrality
        2. PageRank
        3. Алгоритм HITS
        """

