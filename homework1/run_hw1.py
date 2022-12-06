from tasks.task1_old import Task1
from tasks.task2 import Task2
from tasks.task3 import Task3
from preprocessing_utils import get_vk, get_web, get_astro, get_vk_friends, CLEAR_DATA_ROOT
from utils import get_logger

log = get_logger(__name__)


def main():
    log.info('Start Homework1')
    for Task in [
        Task1,
        Task2,
        Task3,
    ]:
        log.info(f'Started {Task.prefix}...')


if __name__ == '__main__':
    main()
