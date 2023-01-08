import os
import random

import networkx as nx
import pandas as pd
import matplotlib
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as Mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from tqdm import tqdm

from homework3.task_defaults import DATA_ROOT, RESULTS_ROOT
from utils import get_logger

matplotlib.use('Agg')
log = get_logger(__name__)


class Task1:
    prefix = 'task1'

    def __init__(self):
        self.imdb = None
        self.gnp = None
        self.sir_gnp = None
        self.gnp_att = None
        self.sir_gnp_att = None

    def run(self):
        self.preprocessing()

        n = len(nx.nodes(self.imdb))
        beta = 0.05
        mu = 0.5

        for graph in [
            self.imdb,
            self.gnp,
            self.gnp_att,
            self.sir_gnp,
            self.sir_gnp_att
        ]:
            model = ep.SIRModel(graph)

            inf_node = random.choice(list(nx.nodes(graph)))
            config = Mc.Configuration()
            config.add_model_parameter('beta', beta)
            config.add_model_parameter('gamma', mu)
            config.add_model_initial_configuration('Infected', [inf_node])
            model.set_initial_status(configuration=config)

            current_iteration = model.iteration()

            while current_iteration['node_count'][1] != 0:
                current_iteration = model.iteration()
                print(current_iteration['node_count'])

            break

    def preprocessing(self):
        self.imdb_preprocessing()
        self.imdb = nx.read_gexf(DATA_ROOT / 'imdb_graph.gexf')
        n = len(nx.nodes(self.imdb))
        d_sum = sum([d[1] for d in list(nx.degree(self.imdb))]) / 2
        avg_d = 2 * d_sum / n

        self.gnp_preprocessing(n, avg_d / n)
        self.gnp = nx.read_gexf(DATA_ROOT / 'gnp.gexf')

        self.gnp_att_preprocessing(n, int(d_sum / (n - 1)))
        self.gnp_att = nx.read_gexf(DATA_ROOT / 'gnp_att.gexf')

        self.from_txt_to_gexf('SIR_Gnp.txt')
        self.sir_gnp = nx.read_gexf(DATA_ROOT / 'SIR_Gnp.gexf')

        self.from_txt_to_gexf('SIR_preferential_attachment.txt')
        self.sir_gnp_att = nx.read_gexf(DATA_ROOT / 'SIR_preferential_attachment.gexf')

    @staticmethod
    def imdb_preprocessing():
        """
        Getting the graph with names instead id based on imdb_actor_edges.tsv graph
        """

        if os.path.isfile(DATA_ROOT / 'imdb_graph.gexf'):
            log.info("The IMDB graph is already exist, skip preprocessing")
            return

        graph = nx.Graph(name='imdb')

        edges = pd.read_csv(DATA_ROOT / 'imdb_actor_edges.tsv', sep='\t')
        key = pd.read_csv(DATA_ROOT / 'imdb_actors_key.tsv', sep='\t', encoding='unicode_escape')

        groups = key.groupby('ID')
        for _, row in tqdm(edges.iterrows(), total=len(edges)):
            if row['num_movies'] < 2:
                continue

            name1 = groups.get_group(row['actor1'])['name'].iloc[0]
            name2 = groups.get_group(row['actor2'])['name'].iloc[0]
            graph.add_edge(name1, name2)

        nx.write_gexf(graph, DATA_ROOT / 'imdb_graph.gexf')

    @staticmethod
    def gnp_preprocessing(n, p):
        """
        Getting the random graph G(n, p)
        """
        if os.path.isfile(DATA_ROOT / 'gnp.gexf'):
            log.info("The Gnp graph is already exist, skip preprocessing")
            return

        graph = nx.gnp_random_graph(n, p, seed=42)
        graph = nx.Graph(graph, name='gnp')
        nx.write_gexf(graph, DATA_ROOT / 'gnp.gexf')

    @staticmethod
    def gnp_att_preprocessing(n, m):
        """
        Getting the random graph G(n, m) with preferential attachment
        """

        if os.path.isfile(DATA_ROOT / 'gnp_att.gexf'):
            log.info("The Gnp_att graph is already exist, skip preprocessing")
            return

        graph = nx.barabasi_albert_graph(n, m, seed=42)
        graph = nx.Graph(graph, name='gnp_att')
        nx.write_gexf(graph, DATA_ROOT / 'gnp_att.gexf')

    @staticmethod
    def from_txt_to_gexf(txt):
        """
        Getting the random graph G(n, m) with preferential attachment
        """
        fname = f'{txt.split(".")[0]}'
        if os.path.isfile(DATA_ROOT / f'{fname}.gexf'):
            log.info(f"The {fname} graph is already exist, skip preprocessing")
            return

        graph = nx.Graph(name=fname)
        with open(DATA_ROOT / txt, 'r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                node1, node2 = line.split()
                graph.add_edge(node1, node2)

        nx.write_gexf(graph, DATA_ROOT / f'{fname}.gexf')
