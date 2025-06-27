import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


in_path = "../master-thesis/glaucoma_oct_data/retina-oct-glaucoma-NPY/retina-oct-glaucoma-NPY"
out_path = "data3d/"

# dataset split documentation in master-thesis project
print("Loading train and test indeces...")
print(os.getcwd())
train_indeces = np.loadtxt("../master-thesis/data/train_data_indeces.npy").astype(np.int64)
test_indeces = np.loadtxt("../master-thesis/data/test_data_indeces.npy").astype(np.int64)

# get patient IDs and pathologies from file names in folder
pathologies = []
patient_ids = []
filenames = []
for filename in os.listdir(in_path):
    if filename.endswith('.npy'):
        # Remove the .npy extension if needed
        name = filename[:-4] if filename.endswith('.npy') else filename
        parts = name.split('-')

        # Extract metadata
        pathologies.append(parts[0]) # Normal or POAG
        patient_ids.append(parts[1]) # The number after
        filenames.append(name) # The full name without extension

# make a dataframe of the loaded information
patient_id = pd.Series(patient_ids, copy=False, name='patient_id')
pathology = pd.Series(pathologies, copy=False, name = 'pathology')
filename = pd.Series(filenames, copy=False, name='filename')
dataframe = pd.concat([patient_id, pathology, filename], axis=1)
dataframe.loc[dataframe['pathology'] == 'Normal', 'pathology'] = 'negative'
dataframe.loc[dataframe['pathology'] == 'POAG', 'pathology'] = 'positive'

# number of images in dataset; sanity check
print("Number of images in dataset: ", dataframe.shape[0])
if dataframe['pathology'].nunique() > 2:
    raise Exception("More than two classes found in the filtered dataset.")

#create data split based on same indeces as model training (minus validation)
df_train_b4 = dataframe.iloc[train_indeces]
SG_Kfold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
cv_splits = SG_Kfold.split(df_train_b4, df_train_b4['pathology'], df_train_b4['patient_id'])
train_indeces, val_indeces = next(cv_splits)
np.savetxt("train_minus_val_data_indeces.npy", train_indeces)
np.savetxt("val_data_indeces.npy", val_indeces)

df_train = df_train_b4.iloc[train_indeces]
df_val = df_train_b4.iloc[val_indeces]
df_test = dataframe.iloc[test_indeces]

# loop through datframes and save images in the same folder structure as original GANterfactual implementation
for data, df_name in zip([df_train, df_val, df_test],["train", "val", "test"]):
    for label in ["negative", "positive"]:
        df_label = data[data["pathology"] == label]
        print(f"Number of images in {label} {df_name} dataset: ", df_label.shape[0])
        image_names = df_label['filename'].tolist() 
        folder_out_path = os. path.join(out_path, df_name, label)
        os.makedirs(folder_out_path, exist_ok=True)
        for image in image_names:
            imagenpy = image + '.npy' 
            image_in_path = os.path.join(in_path, imagenpy)
            assert (os.path.isfile(image_in_path))
            image_out_path = os.path.join(folder_out_path, image)
            arr = np.load(image_in_path)
            np.save(image_out_path, arr)
            # dim = (32,64,32)
            # im = im.resize(dim)
