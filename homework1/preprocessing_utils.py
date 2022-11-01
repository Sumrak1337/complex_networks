import os
import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from defaults import DATA_ROOT
from utils import get_logger

os.makedirs('clear_data', exist_ok=True)
os.makedirs('results', exist_ok=True)
CLEAR_DATA_ROOT = Path(__file__).parent.absolute() / 'clear_data'
RESULTS_ROOT = Path(__file__).parent.absolute() / 'results'

log = get_logger(__name__)


def read_txt(txt):
    g = nx.Graph()
    with open(DATA_ROOT / txt, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        if '#' in line:
            continue

        source, target = line.split()
        g.add_node(source)
        g.add_node(target)
        g.add_edge(source, target)
    return g


def get_vk():
    name = 'vk.gexf'
    if (CLEAR_DATA_ROOT / name).is_file():
        log.info(f'{name} is already exist, skipping creating data files!')
        return

    log.info(f'Getting {name}...')
    vk = pd.read_csv(DATA_ROOT / 'vk.csv')
    g = nx.Graph()
    for _, row in tqdm(vk.iterrows(), total=vk.shape[0]):
        g.add_edge(row.u, row.v)
    nx.write_gexf(g, CLEAR_DATA_ROOT / name)


def get_vk_friends():
    name = 'vk_friends_graph.gexf'
    if (CLEAR_DATA_ROOT / name).is_file():
        log.info(f'{name} is already exist, skipping creating data files!')
        return

    log.info(f'Getting {name}...')
    nx.write_gexf(nx.to_undirected(nx.read_gexf(DATA_ROOT / name)), CLEAR_DATA_ROOT / name)


def get_web():
    name = 'web-Google'
    if (CLEAR_DATA_ROOT / f'{name}.gexf').is_file():
        log.info(f'{name}.gexf is already exist, skipping creating data files!')
        return

    log.info(f'Getting {name}.gexf...')
    g = read_txt(f'{name}.txt')
    nx.write_gexf(g, CLEAR_DATA_ROOT / f'{name}.gexf')


def get_astro():
    name = 'CA-AstroPh'
    if (CLEAR_DATA_ROOT / f'{name}.gexf').is_file():
        log.info(f'{name}.gexf is already exist, skipping creating data files!')
        return

    log.info(f'Getting {name}.gexf...')
    g = read_txt(f'{name}.txt')
    nx.write_gexf(g, CLEAR_DATA_ROOT / f'{name}.gexf')
