import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from matplotlib import cm

from utils import AbstractTask, get_logger
from homework1.task_defaults import DATA_ROOT, RESULTS_ROOT

colors = list(mcolors.TABLEAU_COLORS)
log = get_logger(__name__)


class Task3(AbstractTask):
    prefix = 'task3'

    def __init__(self):
        self.G = nx.read_gexf(DATA_ROOT / 'vk_friends_graph.gexf')
        self.G = nx.to_undirected(self.G)

    def run(self):
        lcc = max(nx.connected_components(self.G), key=len)
        subgraph = nx.subgraph(self.G, lcc)

        pos = nx.spring_layout(subgraph,
                               seed=42)
        cmap = plt.get_cmap('jet')

        hub, autorities = nx.hits(subgraph)

        def get_graph_visualization(cntr, tag):
            norm = mcolors.Normalize(vmin=min(cntr.values()), vmax=max(cntr.values()))
            scmp = cm.ScalarMappable(norm=norm, cmap=cmap)
            color = [scmp.to_rgba(v, alpha=0.5) for v in cntr.values()]
            norm_centrality = [v * 1 / max(cntr.values()) for v in cntr.values()]
            node_size = [v * 1.5e3 for v in norm_centrality]

            plt.figure(figsize=(16, 10))
            plt.title(f'Visualization of {tag}')
            nx.draw_networkx_nodes(subgraph,
                                   pos=pos,
                                   node_color=color,
                                   node_size=node_size)
            nx.draw_networkx_edges(subgraph,
                                   pos=pos,
                                   alpha=0.2)
            nx.draw_networkx_labels(subgraph,
                                    pos=pos,
                                    font_size=9)
            plt.colorbar(scmp)
            plt.tight_layout()
            plt.savefig(RESULTS_ROOT / f'{tag}_vis.png')
            plt.close('all')

        for centrality, c_tag in zip([nx.degree_centrality(subgraph),
                                      nx.closeness_centrality(subgraph),
                                      nx.betweenness_centrality(subgraph),
                                      nx.eigenvector_centrality(subgraph),
                                      self.decay_centrality(subgraph, delta=0.2),
                                      self.decay_centrality(subgraph, delta=0.5),
                                      self.decay_centrality(subgraph, delta=0.8),
                                      nx.pagerank_alg.pagerank(subgraph),
                                      hub,
                                      autorities],
                                     ['degree_centrality',
                                      'closeness_centrality',
                                      'betweenness_centrality',
                                      'eigenvector_centrality',
                                      'decay_centrality_02',
                                      'decay_centrality_05',
                                      'decay_centrality_08',
                                      'page_rank',
                                      'hits_hub',
                                      'hits_autorities'
                                      ]):
            self.get_top_10(centrality, c_tag)
            get_graph_visualization(centrality, c_tag)

    @staticmethod
    def get_top_10(metric, tag):
        lst = sorted(metric.items(), key=lambda x: x[1], reverse=True)

        log.info(f'{tag}:')
        for name, value in lst[:10]:
            print(f'{name:20}: {value:.4f}')

    @staticmethod
    def decay_centrality(subgraph, delta):
        dc = {}
        nodes = nx.nodes(subgraph)
        for node1 in nodes:
            dc[node1] = 0
            for node2 in nx.nodes(subgraph):
                if node1 != node2:
                    dc[node1] += pow(delta, nx.shortest_path_length(subgraph, source=node1, target=node2))
        return dc
