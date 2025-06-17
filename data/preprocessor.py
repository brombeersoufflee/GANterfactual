from PIL import Image
import os
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pydicom import dcmread
import numpy as np
from skimage.transform import resize

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--in', required=True, help='input folder')
#ap.add_argument('-o', '--out', required=True, help='output folder')
ap.add_argument('-t', '--test', required=True, help='proportion of images used for test')
ap.add_argument('-v', '--validation', required=True, help='proportion of images used for validation')
ap.add_argument('-d', '--dimension', required=True, help='new dimension for files')
args = vars(ap.parse_args())


# TODO: modify dim paramater for own dataset, might be easier than rewriting the code to match my own dataloading (watch out with duplicate patient records)
def preprocess(in_path, out_path, test_size, val_size, dim):
    
    # get labels from csv file
    df_img_lbl = pd.read_csv("stage_2_detailed_class_info.csv")
    df_img_lbl = df_img_lbl.drop_duplicates(subset='patientId')
    print("Number of images in dataset: ", df_img_lbl.shape[0])
    df_img_lbl = df_img_lbl[df_img_lbl["class"] != "No Lung Opacity / Not Normal"]
    if df_img_lbl['class'].nunique() > 2:
        raise Exception("More than two classes found in the filtered dataset.")

    le = LabelEncoder() # TODO: Label encoder: ensure class label consistency // currently incorrect labelling
    le.fit(df_img_lbl['class'])
    df_img_lbl['class'] = le.transform(df_img_lbl['class'])
    print("Classes found: ", le.classes_)
    # df_img_lbl = df_img_lbl.set_index('patientId')

    for idx, label in enumerate(["positive", "negative"]):
        df_label = df_img_lbl[df_img_lbl["class"] == idx]
        print(f"Number of images in {label} dataset: ", df_label.shape[0])
        image_names = df_label['patientId'].tolist()  # Get the list of image names
        # df_label1 = df_label["class"].to_dict()  # Convert to dictionary for easier access
        print(f"Number of images in {label} dataset after filtering: ", len(image_names))

        train, test_val = train_test_split(image_names, test_size=test_size + val_size)
        test, val = train_test_split(test_val, test_size=val_size / (test_size + val_size))

        train_path = os. path.join(out_path, 'train', label)
        os.makedirs(train_path, exist_ok=True)
        test_path = os.path.join(out_path, 'test', label)
        os.makedirs(test_path, exist_ok=True)
        val_path = os.path.join(out_path, 'validation', label)
        os.makedirs(val_path, exist_ok=True)

        resize_im(in_path, train_path, train, dim)
        resize_im(in_path, test_path, test, dim)
        resize_im(in_path, val_path, val, dim)


def resize_im(in_path, out_path, images, dim):
    for image in images:
        imagedcm = image + '.dcm' 
        image_in_path = os.path.join(in_path, imagedcm)
        # print("Processing image: ", image_in_path)
        assert (os.path.isfile(image_in_path))
        imagepng = image + '.png'
        image_out_path = os.path.join(out_path, imagepng)

        with dcmread(image_in_path) as ds:
                data = ds.pixel_array
                im = Image.fromarray(data)
                im_resized = im.resize((dim, dim))
                im_resized.save(image_out_path, quality=100)

cwd = os.getcwd()
print(cwd)
preprocess(args['in'], os.path.join('.'), float(args['test'])/100, float(args['validation'])/100, int(args['dimension']))

