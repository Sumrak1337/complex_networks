import networkx as nx
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utils import AbstractTask
from homework2.task_defaults import CLEAR_DATA_ROOT

# TODO: change 1500 rows


class Task3(AbstractTask):
    prefix = 'task3'

    def __init__(self):
        self.random_graph = None
        self.snowball_graph = None

    def run(self):
        self.preprocessing()

        self.random_graph = nx.read_gexf(CLEAR_DATA_ROOT / 'imdb_random.gexf')
        self.snowball_graph = nx.read_gexf(CLEAR_DATA_ROOT / 'imdb_snowball.gexf')

    def preprocessing(self):
        output_path_random = CLEAR_DATA_ROOT / 'imdb_random.gexf'
        output_path_snowball = CLEAR_DATA_ROOT / 'imdb_snowball.gexf'
        if os.path.isfile(output_path_random) and os.path.isfile(output_path_snowball):
            print('All GEXF files are exist, skip preprocessing')
            return

        # Generate random graph
        self.get_csv_actors()
        self.get_gexf_random_actors(output_path_random)

        # Generate snowball graph
        self.get_gexf_snowball_actors(output_path_snowball)

    @staticmethod
    def get_csv_actors():
        output_path_random_csv = CLEAR_DATA_ROOT / 'actors_random.csv'
        if os.path.isfile(output_path_random_csv):
            print(f'Random graph CSV is already exist, skip')
            return

        data = pd.read_csv(CLEAR_DATA_ROOT / 'data_actors.tsv', sep='\t')
        actors = pd.DataFrame(columns=data.columns)

        ids = []
        rs = np.random.RandomState(42)

        while len(actors) < 1000:
            idx = rs.randint(1, 1500)

            current_actor = data.iloc[idx]

            professions = current_actor.primaryProfession
            if not isinstance(professions, str):
                continue

            professions = professions.split(',')
            if ('actor' in professions or 'actress' in professions) and idx not in ids:
                actors = pd.concat([actors, current_actor.to_frame().T], ignore_index=True)
                ids.append(idx)

        actors.to_csv(CLEAR_DATA_ROOT / f'actors_random.csv', index=False)

    @staticmethod
    def get_gexf_random_actors(path):
        if os.path.isfile(path):
            print('Random graph GEXF file is already exist, skip')
            return

        graph = nx.Graph()
        data = pd.read_csv(CLEAR_DATA_ROOT / 'actors_random.csv')

        for i, row in tqdm(data.iterrows(), total=data.shape[0]):
            current_id = row.nconst
            graph.add_node(current_id)
            film_ids = row.knownForTitles
            if not isinstance(film_ids, str):
                continue
            film_ids = film_ids.split(',')

            for film_id in film_ids:
                for j in range(i + 1, len(data)):
                    new_row = data.iloc[j]
                    new_films = new_row.knownForTitles.split(',')
                    if film_id in new_films:
                        new_id = new_row.nconst
                        graph.add_node(new_id)
                        graph.add_edge(current_id, new_id)

        nx.write_gexf(graph, path)

    @staticmethod
    def get_gexf_snowball_actors(path):
        data = pd.read_csv(CLEAR_DATA_ROOT / 'data_actors.tsv', sep='\t')
        graph = nx.Graph()
        some_person_index = 205
        actors_indices = [some_person_index]

        def check_films(list1, list2):
            # TODO: probably, can be simplify
            for l1 in list1:
                for l2 in list2:
                    if l1 == l2:
                        return True
            return False

        for actor in actors_indices:
            current_id = data.iloc[actor].nconst
            graph.add_node(current_id)
            current_films = data.iloc[actor].knownForTitles.split(',')

            for i, row in data.head(1500).iterrows():
                new_id = row.nconst
                new_films = row.knownForTitles.split(',')
                if check_films(current_films, new_films) and current_id != new_id:
                    graph.add_node(new_id)
                    graph.add_edge(current_id, new_id)

                    if i not in actors_indices:
                        actors_indices.append(i)

                    if len(nx.nodes(graph)) == 1000:
                        break

            if len(nx.nodes(graph)) == 1000:
                break

        nx.write_gexf(graph, path)
