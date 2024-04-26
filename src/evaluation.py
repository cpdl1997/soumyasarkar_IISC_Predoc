import pandas as pd
import numpy as np
# import matplotplib as plt
import os
import math

import logging
exp_name = "evaluation"
log_file_name = "Log_" + exp_name + ".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
# fh.setLevel(logging.INFO)
logger.addHandler(fh)

from model import Model1
from analysis import dataset_analysis
from preprocess import dataset_preprocess

#-------------------------------------Sample equally from all directors to create test and train-------------------------------------
def split_test_train(df, genre, train_split=0.7):
    columns = df.columns.tolist()
    columns_x =[]
    columns_y =[]
    for i in columns:
        if i in genre.columns:
            columns_y.append(i)
        elif i=='title_year':
            columns_y.append(i)
        else:
            columns_x.append(i)

    x_train = pd.DataFrame(columns=columns_x)
    y_train = pd.DataFrame(columns=columns_y)
    x_test = pd.DataFrame(columns=columns_x)
    y_test = pd.DataFrame(columns=columns_y)

    #-------------------------------------Director-wise Addition-------------------------------------
    all_directors = df['director_name'].unique()
    for i in all_directors:
        df_dir = df[df['director_name']==i]
        # logger.info("\n-------------------------------------df for director {}-------------------------------------".format(i))
        # logger.info(df_dir.head(10).to_markdown())
        
        num_train = math.floor(train_split*len(df_dir))
        df_train = df_dir.iloc[:num_train]
        df_test = df_dir.iloc[num_train:]
        # logger.info("\n-------------------------------------df_train for director {}-------------------------------------".format(i))
        # logger.info(df_train.head().to_markdown())
        # logger.info("\n-------------------------------------df_test for director {}-------------------------------------".format(i))
        # logger.info(df_test.head().to_markdown())
        # logger.info("\n-------------------------------------df_train for director {} after dropping y columns-------------------------------------".format(i))
        # logger.info(df_train.drop(columns_y, axis=1).head().to_markdown())
        # logger.info("\n-------------------------------------df_test for director {} after dropping x columns-------------------------------------".format(i))
        # logger.info(df_test.drop(columns_x, axis=1).head().to_markdown())
        x_train = pd.concat([x_train, df_train.drop(columns_y, axis=1)], ignore_index=True)
        y_train = pd.concat([y_train, df_train.drop(columns_x, axis=1)], ignore_index=True)
        x_test = pd.concat([x_test, df_test.drop(columns_y, axis=1)], ignore_index=True)
        y_test = pd.concat([y_test, df_test.drop(columns_x, axis=1)], ignore_index=True)

    # logger.info("\n-------------------------------------x_train-------------------------------------")
    # logger.info(x_train.head().to_markdown())
    # logger.info("\n-------------------------------------y_train-------------------------------------")
    # logger.info(y_train.head().to_markdown())
    # logger.info("\n-------------------------------------x_test-------------------------------------")
    # logger.info(x_test.head().to_markdown())
    # logger.info("\n-------------------------------------y_test-------------------------------------")
    # logger.info(y_test.head().to_markdown())

    return x_train, y_train, x_test, y_test


def run(_learning_rate=0.01):
    dataset_name = "p1_movie_metadata.csv"
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name)
    df = pd.read_csv(dataset_path)
    df = dataset_preprocess.remove_nan(df=df) #Drop NaN values before proceeding with statistics
    # dataset_analysis.run(df=df)
    df, genre = dataset_preprocess.expand_genre(df=df)
    train_split = 0.8
    x_train, y_train, x_test, y_train = split_test_train(df, genre, train_split)
    model = Model1.Model1(_in_features = len(df.columns)-2, _out_features = len(genre.columns), _learning_rate = _learning_rate,)
    


if __name__=="__main__":
    _learning_rate = 0.01
    run(_learning_rate)