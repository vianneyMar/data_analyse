import numpy as np
import h5py
import pandas as pd

# pd.set_option('display.max_rows', None) 
# pd.set_option('display.max_columns', None) 

# list features 


#set path to datasets

class Dataloader(object):
    def __init__(self, dataset_directory_path = '/mnt/c/Users/w133572/Downloads/laz_vianney_math/challenge/'):
        self.dataset_directory_path = dataset_directory_path

    def load_data(self, h5_path, shape):
        #shape tuple (x,y,z)
        file = h5py.File(self.dataset_directory_path + h5_path, 'r')
        #load h5 file 
        data_content = file.get('my_data')
        #cast intot np array
        data_content= np.array(data_content)
        # print(data_content.shape)
        #reshape at wished size
        data_content= np.reshape(data_content,shape)
        #return numpy dim 3 
        return data_content
    
    def format_with_pelvis(self, dataframe):
        new_df = pd.DataFrame()
        data_dict = {}

        for key in dataframe.columns:
            if key.startswith("pelvis"):
                # On garde pelvis_* tel quel
                data_dict[key] = dataframe[key]
            else:
                # Centrage par rapport au bassin
                if "_x" in key:
                    data_dict[key] = dataframe[key] - dataframe["pelvis_x"]
                elif "_y" in key:
                    data_dict[key] = dataframe[key] - dataframe["pelvis_y"]
                elif "_z" in key:
                    data_dict[key] = dataframe[key] - dataframe["pelvis_z"]
                else:
                    data_dict[key] = dataframe[key]  # au cas où

        new_df = pd.DataFrame(data_dict)
        new_df.drop(columns=["pelvis_x","pelvis_y","pelvis_z"], inplace=True)
    
        return new_df    

    def remove_outliers_iqr(self, df, factor=1.5, replace_with="median"):
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

    def load_batch(self, data, batch_indice, features):
        return pd.DataFrame(data[batch_indice], features)


# df_mocap = pd.DataFrame(mocap[0], columns=features_mocap)
# df_insole = pd.DataFrame(insoles[0], columns=features_insoles)  # ici c'était mocap[0] par erreur
# df_concat = pd.concat([df_insole, df_mocap], axis=1)
# print(df_concat)  # Résultat attendu : (100, 179)

