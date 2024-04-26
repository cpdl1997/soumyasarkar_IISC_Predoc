import pandas as pd
import numpy as np
# import matplotplib as plt
import os

import logging
exp_name = "data_visualization_log"
log_file_name = "Log_" + exp_name + ".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "..", "logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
# fh.setLevel(logging.INFO)
logger.addHandler(fh)



def dataset_details(dataset_path):

    df = pd.read_csv(dataset_path)
    #Generic details
    logger.info("-------------------------------------A Brief Look at the Dataset-------------------------------------")
    logger.info("Total Number of Entries = {} and number of columns = {}".format(len(df), len(df.columns)))
    logger.info(df.head().to_markdown())
    logger.info("-------------------------------------More Details abut Dataset-------------------------------------")

    #NaN details
    logger.info("\n-------------------------------------NaN Values-------------------------------------")
    for i in df.columns:
        logger.info("Number of NaN values in column {} = {}".format(i, df[i].isna().sum()))
    logger.info("We will drop all NaN values.")
    df = df.dropna(how='any',axis=0)
    logger.info("Total Number of Entries remaining = {}".format(len(df)))

    #Unique entries details
    logger.info("\n-------------------------------------Unique Values-------------------------------------")
    for i in df.columns:
        logger.info("Number of unique values in column {} = {}".format(i, len(df[i].unique())))

    # #Unique entries details
    # logger.info("\n-------------------------------------Director-wise Stats-------------------------------------")
    # all_directors = df['director_name'].unique()
    # for i in all_directors:
    #     logger.info("\n-------------------------------------{}-------------------------------------".format(i))
    #     df_dir = df[df['director_name']==i]
    #     logger.info("Number of Movies: {}".format(len(df_dir['movie_title'].unique())))
    #     logger.info("Average duration of Movie: {}".format(sum(df_dir['duration'])/len(df_dir)))
    #     logger.info("Languages directed in: {}".format(df_dir['language'].unique()))
    #     logger.info("Average Budget for a Movie: {} million".format(sum(df_dir['budget'])/(len(df_dir)*100000)))
    #     logger.info("Average Gross Income for a Movie: {} million".format(sum(df_dir['gross'])/(len(df_dir)*100000)))
    #     max_budget_row = df_dir.loc[df_dir.index[df_dir['budget'] == max(df_dir['budget'])]] #gives a dataframe for the row with maximum budget value
    #     logger.info("Highest Budget Movie: \"{}\" with a budget of {} million and gross income of {} million.".format(max_budget_row.iloc[0]['movie_title'][:-1], float(max_budget_row.iloc[0]['budget'])/100000, float(max_budget_row.iloc[0]['gross'])/100000)) #Using [:-1] to remove last unknown character

    logger.info("\n-------------------------------------Split on Column 'Genres'-------------------------------------")
    
    df_left_old = df.iloc[:, :df.columns.get_loc('genres')] #All columns on left of genres
    df_right_old = df.iloc[:, df.columns.get_loc('genres')+1:] #All columns on right of genres
    # df_genres = df['genres'].str.split('|', expand=True)
    
    df['genres'] = df['genres'].str.split('|') #Split into single string value with spaces instead of |
    df = df.explode('genres') #Split genre column into multiple rows with each row taking one value of genre
    df_left = df.iloc[:, :df.columns.get_loc('genres')]
    df_right = df.iloc[:, df.columns.get_loc('genres')+1:]
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

    logger.info(df.head(10).to_markdown())
    
    

    



def run():
    
    dataset_name = "p1_movie_metadata.csv"
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "dataset", dataset_name)
    dataset_details(dataset_path=dataset_path)




if __name__=="__main__":
    run()