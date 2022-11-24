from utils import get_logger
from homework2.tasks.task1 import Task1
from homework2.tasks.task2 import Task2
from homework2.tasks.task3 import Task3

log = get_logger(__name__)


def main():
    for Task in [
        Task1,
        Task2,
        Task3,
    ]:
        Task().run()


if __name__ == '__main__':
    main()
