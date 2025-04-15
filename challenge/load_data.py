import numpy as np
import h5py
import pandas as pd

# pd.set_option('display.max_rows', None) 
# pd.set_option('display.max_columns', None) 

features_insoles = [
    "left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_7", "left_8", "left_9", "left_10",
    "left_11", "left_12", "left_13", "left_14", "left_15", "left_16",
    "left_acc_x", "left_acc_y", "left_acc_z",
    "left_ang_x", "left_ang_y", "left_ang_z",
    "left_force", "left_center_x", "left_center_y",
    "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_7", "right_8", "right_9", "right_10",
    "right_11", "right_12", "right_13", "right_14", "right_15", "right_16",
    "right_acc_x", "right_acc_y", "right_acc_z",
    "right_ang_x", "right_ang_y", "right_ang_z",
    "right_force", "right_center_x", "right_center_y"
]
features_mocap = []
for i in range(1, 43):
    features_mocap.append(f"p{i}_x")
    features_mocap.append(f"p{i}_y")
    features_mocap.append(f"p{i}_z")
features_mocap.append("pelvis_x")
features_mocap.append("pelvis_y")
features_mocap.append("pelvis_z")

datapath = '/mnt/c/Users/w133572/Downloads/laz_vianney_math/challenge/'

file = h5py.File(datapath + 'train_mocap.h5', 'r')
mocap = file.get('my_data')
mocap= np.array(mocap)
mocap= np.reshape(mocap,(6938,100,129))

file = h5py.File(datapath + 'train_insoles.h5', 'r')
insoles = file.get('my_data')
insoles= np.array(insoles)
insoles= np.reshape(insoles,(6938,100,50))

df_mocap = pd.DataFrame(mocap[0], columns=features_mocap)
df_insole = pd.DataFrame(insoles[0], columns=features_insoles)  # ici c'était mocap[0] par erreur

df_concat = pd.concat([df_insole, df_mocap], axis=1)
# print(df_concat)  # Résultat attendu : (100, 179)

def format_with_pelvis(mocap_dataset):
    new_df = pd.DataFrame()
    data_dict = {}

    for key in mocap_dataset.columns:
        if key.startswith("pelvis"):
            # On garde pelvis_* tel quel
            data_dict[key] = mocap_dataset[key]
        else:
            # Centrage par rapport au bassin
            if "_x" in key:
                data_dict[key] = mocap_dataset[key] - mocap_dataset["pelvis_x"]
            elif "_y" in key:
                data_dict[key] = mocap_dataset[key] - mocap_dataset["pelvis_y"]
            elif "_z" in key:
                data_dict[key] = mocap_dataset[key] - mocap_dataset["pelvis_z"]
            else:
                data_dict[key] = mocap_dataset[key]  # au cas où

    new_df = pd.DataFrame(data_dict)
    new_df.drop(columns=["pelvis_x","pelvis_y","pelvis_z"], inplace=True)
 
    return new_df 

print("classic")
print(df_mocap)

print("reworked")
formated_with_pelvis = format_with_pelvis(df_mocap)

def remove_outliers_iqr(df, factor=1.5, replace_with="median"):
    df_clean = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outliers = (df[col] < lower) | (df[col] > upper)

            if replace_with == "median":
                df_clean[col] = df[col].mask(outliers, df[col].median())
            elif replace_with == "mean":
                df_clean[col] = df[col].mask(outliers, df[col].mean())
            elif replace_with == "zero":
                df_clean[col] = df[col].mask(outliers, 0)
            elif replace_with == "nan":
                df_clean[col] = df[col].mask(outliers, np.nan)
            else:  # keep outliers
                pass
    return df_clean

df_filtered = remove_outliers_iqr(formated_with_pelvis)
print("removed outliers")
print(df_filtered)