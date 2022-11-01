import networkx as nx

from tasks.task1 import Task1
from preprocessing_utils import get_vk, get_web, get_astro, get_vk_friends, CLEAR_DATA_ROOT
from utils import get_logger

# TODO: add milestones (can be useful)

log = get_logger(__name__)


def preprocessing():
    get_vk()
    get_vk_friends()
    get_web()
    get_astro()


def main():
    log.info('Start preprocessing...')
    preprocessing()
    log.info('End preprocessing.')
    list_of_files = ['CA-AstroPh',
                     'vk',
                     'vk_friends_graph',
                     'web-Google']

    for file in list_of_files:
        # if file != 'vk_friends_graph':
        #     continue
        log.info(f'{file}:')
        g = nx.read_gexf(CLEAR_DATA_ROOT / f'{file}.gexf')
        for Task in [Task1,
                     ]:
            log.info(f'Started {Task.prefix}...')
            task = Task(g, file)
            task.run()

        # break


if __name__ == '__main__':
    main()
