import pandas as pd
import numpy as np
# import matplotplib as plt
import os
from sklearn.preprocessing import LabelEncoder

import logging
exp_name = "dataset_preprocess"
log_file_name = "Log_" + exp_name + ".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "..", "logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
# fh.setLevel(logging.INFO)
logger.addHandler(fh)


def remove_nan(df):
    #NaN details
    logger.info("\n-------------------------------------NaN Values-------------------------------------")
    for i in df.columns:
        logger.info("Number of NaN values in column {} = {}".format(i, df[i].isna().sum()))
    logger.info("We will drop all NaN values.")
    df = df.dropna(how='any',axis=0)
    logger.info("Total Number of Entries remaining = {}".format(len(df)))
    return df



#-------------------------------------Split on Column 'Genres'-------------------------------------
def expand_genre(df):
    df['genres'] = df['genres'].str.split('|') #Split into single string value with spaces instead of |
    df = df.explode('genres') #Split genre column into multiple rows with each row taking one value of genre
    df_left = df.iloc[:, :df.columns.get_loc('genres')] #All columns on left of genres with
    df_right = df.iloc[:, df.columns.get_loc('genres')+1:] #All columns on right of genres initially
    one_hot = pd.get_dummies(df['genres']) #Create one hot encoding based on 'genres' column
    df = pd.concat([df_left, one_hot, df_right], axis=1) #Concat to create the dataframe with one hot encoding

    #convert to multi-hot encoding of genre for each movie
    aggregation_functions = {}
    for j in df.columns:
        if j!='movie_title':
            if j in one_hot.columns:
                aggregation_functions[j] = 'sum'
            else:
                aggregation_functions[j] = 'first'
    #reindex in order to get same column ordering as original dataframe
    #as_index=False used to stop movie_title from becoming the index
    #sort=False in order to get same row ordering as original dataframe
    df = df.groupby('movie_title', as_index=False, sort=False).aggregate(aggregation_functions).reindex(columns=df.columns) #https://stackoverflow.com/questions/46826773/how-can-i-merge-rows-by-same-value-in-a-column-in-pandas-with-aggregation-func

    # logger.info(df.head(10).to_markdown())
    
    return df, one_hot



#Drop these two columns which won't contribute to result
def drop_extra_columns(df):
    list_of_columns_to_drop = ['plot_keywords', 'movie_imdb_link']
    df = df.drop(list_of_columns_to_drop, axis=1)
    return df



#Returns dictionary of continuous and categorical features for input
def column_type_dict_input():
    continuous = ['num_critic_for_reviews',
                  'duration',
                  'director_facebook_likes',
                  'actor_3_facebook_likes',
                  'actor_1_facebook_likes',
                  'gross',
                  'num_voted_users',
                  'cast_total_facebook_likes',
                  'facenumber_in_poster',
                  'num_user_for_reviews',
                  'budget',
                  'actor_2_facebook_likes',
                  'imdb_score',
                  'aspect_ratio',
                  'movie_facebook_likes'
                  ]
    categorical = ['color',
                   'director_name',
                   'actor_2_name',
                   'actor_1_name',
                   'movie_title',
                   'actor_3_name',
                   'language',
                   'country',
                   'content_rating',
                   ]
    dict = {}
    dict['continuous'] = continuous
    dict['categorical'] = categorical
    return dict



#Do label encoding based on column type (continuous or categorical)
def label_encode(df, column_type):
    le = LabelEncoder()
    for i in df.columns:
        if i in column_type['categorical']:
            df[i] = le.fit_transform(df[i])
    return df
