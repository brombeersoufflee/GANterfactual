from PIL import Image
import os
import argparse
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--in', required=True, help='input folder')
#ap.add_argument('-o', '--out', required=True, help='output folder')
ap.add_argument('-t', '--test', required=True, help='proportion of images used for test')
ap.add_argument('-v', '--validation', required=True, help='proportion of images used for validation')
ap.add_argument('-d', '--dimension', required=True, help='new dimension for files')
args = vars(ap.parse_args())


def preprocess(in_path, out_path, test_size, val_size, dim):
    # dirs = os.listdir(in_path)

    # for label in dirs:
    #     # print(label)
    #     label_path = os.path.join(in_path, label)
    #     # assert (os.path.isdir(label_path))

    train_path = os.path.join(out_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    test_path = os.path.join(out_path, 'test')
    os.makedirs(test_path, exist_ok=True)
    val_path = os.path.join(out_path, 'validation')
    os.makedirs(val_path, exist_ok=True)

    images = os.listdir(in_path)

    train, test_val = train_test_split(images, test_size=test_size + val_size)
    test, val = train_test_split(test_val, test_size=val_size / (test_size + val_size))

    resize(in_path, train_path, train, dim)
    resize(in_path, test_path, test, dim)
    resize(in_path, val_path, val, dim)


def resize(in_path, out_path, images, dim):
    for image in images:
        image_in_path = os.path.join(in_path, image)
        assert (os.path.isfile(image_in_path))
        image_out_path = os.path.join(out_path, image)

        im = Image.open(image_in_path)
        im_resized = im.resize((dim, dim), Image.ANTIALIAS)
        im_resized.save(image_out_path, 'png', quality=100)

cwd = os.getcwd()
print(cwd)
preprocess(args['in'], os.path.join('.'), float(args['test']), float(args['validation']), int(args['dimension']))
