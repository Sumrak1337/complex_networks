import networkx as nx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import AbstractTask
from homework2.task_defaults import CLEAR_DATA_ROOT, RESULTS_ROOT

# TODO: add logging


class Task3(AbstractTask):
    prefix = 'task3'

    def __init__(self):
        self.random_graph = None
        self.snowball_graph = None
        self.full = True

    def run(self):
        if self.full:
            self.preprocessing_of_full_db()
            self.random_graph = nx.read_gexf(CLEAR_DATA_ROOT / 'imdb_random_full.gexf')
            self.snowball_graph = nx.read_gexf(CLEAR_DATA_ROOT / 'imdb_snowball_full.gexf')
        else:
            # self.preprocessing_of_current_db()
            self.random_graph = nx.read_gexf(CLEAR_DATA_ROOT / 'imdb_random_cur.gexf')
            self.snowball_graph = nx.read_gexf(CLEAR_DATA_ROOT / 'imdb_snowball_cur.gexf')

        # Density comparison
        random_den = nx.density(self.random_graph)
        snowball_den = nx.density(self.snowball_graph)
        print(f"Random Graph Density: {random_den:.8f}")
        print(f"Snowball Graph Density: {snowball_den:.8f}")

        # Average shortest path length
        snowball_avg_path_length = nx.average_shortest_path_length(self.snowball_graph)
        random_graph_components = [self.random_graph.subgraph(c).copy() for c in nx.connected_components(self.random_graph)]
        random_avg_path_length = 0
        for subgraph in random_graph_components:
            random_avg_path_length += nx.average_shortest_path_length(subgraph)
        random_avg_path_length /= len(random_graph_components)
        print(f"Random Graph average shortest path length: {random_avg_path_length:.4f}")  # TODO: ask
        print(f"Snowball Graph average shortest path length: {snowball_avg_path_length:.4f}")

        # Density Degree distribution
        random_degree_seq = [degree[1] for degree in nx.degree(self.random_graph)]
        snowball_degree_seq = [degree[1] for degree in nx.degree(self.snowball_graph)]
        rds, rds_counts = np.unique(random_degree_seq, return_counts=True)
        sds, sds_counts = np.unique(snowball_degree_seq, return_counts=True)
        delta = 0.9

        plt.figure(figsize=(16, 9))
        plt.bar(rds - delta / 2, rds_counts, width=delta, label='Random graph')
        plt.bar(sds + delta / 2, sds_counts, width=delta, label='Snowball graph')
        plt.xlabel("Degree")
        plt.ylabel("# of samples")
        plt.title("Density Degree Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_ROOT / 'deg_distr.png')

    def preprocessing_of_full_db(self):
        output_path_random = CLEAR_DATA_ROOT / 'imdb_random_full.gexf'
        output_path_snowball = CLEAR_DATA_ROOT / 'imdb_snowball_full.gexf'
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
            idx = rs.randint(1, len(data))

            current_actor = data.iloc[idx]
            films = current_actor.knownForTitles
            if len(films) > 2:
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
                        graph.add_edge(current_id, new_id)

        nx.write_gexf(graph, path)

    @staticmethod
    def get_gexf_snowball_actors(path):
        if os.path.isfile(path):
            print('Snowball graph GEXF file is already exist, skip')
            return

        data = pd.read_csv(CLEAR_DATA_ROOT / 'data_actors.tsv', sep='\t')
        graph = nx.Graph()
        some_person_index = 167
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
            current_films = data.iloc[actor].knownForTitles.split(',')

            for i, row in tqdm(data.iterrows(), total=len(data)):
                new_id = row.nconst
                new_films = row.knownForTitles.split(',')
                if check_films(current_films, new_films) and current_id != new_id:
                    graph.add_edge(current_id, new_id)

                    if i not in actors_indices:
                        actors_indices.append(i)

                    # check, if new actor played with other actors in graph
                    for j in actors_indices:
                        friend_films = data.iloc[j].knownForTitles.split(',')
                        check_id = data.iloc[j].nconst
                        if check_films(friend_films, new_films) and j != i:
                            graph.add_edge(check_id, new_id)

                    if len(nx.nodes(graph)) == 1000:
                        break
            if len(nx.nodes(graph)) == 1000:
                break

        nx.write_gexf(graph, path)
