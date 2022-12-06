from tasks.task1 import Task1
from tasks.task2 import Task2
from tasks.task3 import Task3
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
        Task().run()


if __name__ == '__main__':
    main()
