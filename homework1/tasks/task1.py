import os
import networkx as nx
import pandas as pd
from tqdm import tqdm

from utils import AbstractTask, get_logger
from homework1.task_defaults import CLEAR_DATA_ROOT

log = get_logger(__file__)


class Task1(AbstractTask):
    prefix = 'task1'

    def __init__(self):
        self.graph = None
        self.graph_astro = None  # CA-AstroPh.txt
        self.graph_vk = None  # vk.csv
        self.graph_vk_friends = None  # vk_friends_graph.gexf
        self.graph_web = None  # web-Google.txt

    def run(self):
        self.preprocessing()

        graph_tags = [
            'CA-AstroPh.gexf',
            # 'vk.gexf',
            'vk_friends_graph.gexf',
            'web-Google.gexf'
        ]
        for graph_tag in graph_tags:
            log.info(f'{graph_tag} reading')
            self.graph = nx.read_gexf(CLEAR_DATA_ROOT / graph_tag)
            if graph_tag != 'web-Google.gexf':
                self.graph = nx.to_undirected(self.graph)

    def preprocessing(self):
        log.info('Start preprocessing')
        astro_name = 'CA-AstroPh'
        vk_name = 'vk'
        web_name = 'web-Google'

        self._get_astro(astro_name)
        self._get_vk(vk_name)
        self._get_web(web_name)

        # self.graph_astro = nx.to_undirected(nx.read_gexf(CLEAR_DATA_ROOT / f'{astro_name}.gexf'))
        # self.graph_vk = nx.to_undirected(nx.read_gexf(CLEAR_DATA_ROOT / f'{vk_name}.gexf'))
        # self.graph_vk_friends = nx.to_undirected(nx.read_gexf(CLEAR_DATA_ROOT / f'{vk_friends_name}.gexf'))
        # self.graph_web = nx.read_gexf(CLEAR_DATA_ROOT / f'{web_name}')
        log.info('End preprocessing')

    @staticmethod
    def _read_txt(txt):
        g = nx.Graph()
        with open(CLEAR_DATA_ROOT / txt, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            if '#' in line:
                continue
            source, target = line.split()
            g.add_edge(source, target)
        return g

    def _get_astro(self, name):
        if self._check_exist_gexf(name):
            return

        g = self._read_txt(f'{name}.txt')
        nx.write_gexf(g, CLEAR_DATA_ROOT / f'{name}.gexf')

    def _get_vk(self, name):
        if self._check_exist_gexf(name):
            return

        vk = pd.read_csv(CLEAR_DATA_ROOT / f'{name}.csv')
        g = nx.Graph()
        for _, row in tqdm(vk.iterrows(), total=vk.shape[0]):
            g.add_edge(row.u, row.v)
        nx.write_gexf(g, CLEAR_DATA_ROOT / f'{name}.gexf')

    def _get_web(self, name):
        if self._check_exist_gexf(name):
            return

        g = self._read_txt(f'{name}.txt')
        nx.write_gexf(g, CLEAR_DATA_ROOT / f'{name}.gexf')

    @staticmethod
    def _check_exist_gexf(name):
        if os.path.isfile(CLEAR_DATA_ROOT / f'{name}.gexf'):
            log.info(f'{name}.gexf is already exist, skip')
            return True
        return False
