from preprocessing_utils import get_vk, get_web, get_astro, get_vk_friends
from utils import get_logger


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

    # for Task in [Task1,
    #              ]:
    #     ...


if __name__ == '__main__':
    main()
