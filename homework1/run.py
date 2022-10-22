from preproc_utils import get_vk, get_web, get_astro, get_vk_friends


# TODO: add logging


def preprocessing():
    print('vk.csv')
    get_vk()
    print('vk_friends')
    get_vk_friends()
    print('web-Google')
    get_web()
    print('AstroPh')
    get_astro()


def main():
    print('start preprocessing...')
    preprocessing()
    print('end preprocessing')

    # for Task in [Task1,
    #              ]:
    #     ...


if __name__ == '__main__':
    main()
