import imageio
import os


def item_to_int(item):
    lst = item.split('_')
    lst = lst[2].split('.')
    return int(lst[0])


def solve():
    # Build GIF
    with imageio.get_writer('new_gif.gif', mode='I') as writer:
        files = [filename for filename in os.listdir('images1/')]
        files.sort(key=item_to_int)
        for file in files:
            image = imageio.imread('images1/' + file)
            writer.append_data(image)

    print('GIF created')


if __name__ == '__main__':
    solve()
