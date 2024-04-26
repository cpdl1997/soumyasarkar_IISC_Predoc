import pandas as pd
import numpy as np
# import matplotplib as plt
import os

import logging
exp_name = "dataset_analysis"
log_file_name = "Log_" + exp_name + ".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",  "..", "logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
# fh.setLevel(logging.INFO)
logger.addHandler(fh)



def run(df):

    #Generic details
    logger.info("-------------------------------------A Brief Look at the Dataset-------------------------------------")
    logger.info("Total Number of Entries = {} and number of columns = {}".format(len(df), len(df.columns)))
    logger.info(df.head().to_markdown())
    logger.info("-------------------------------------More Details abut Dataset-------------------------------------")

    #Unique entries details
    logger.info("\n-------------------------------------Unique Values-------------------------------------")
    for i in df.columns:
        logger.info("Number of unique values in column {} = {}".format(i, len(df[i].unique())))

    #Unique entries details
    logger.info("\n-------------------------------------Director-wise Stats-------------------------------------")
    all_directors = df['director_name'].unique()
    for i in all_directors:
        logger.info("\n-------------------------------------{}-------------------------------------".format(i))
        df_dir = df[df['director_name']==i]
        logger.info("Number of Movies: {}".format(len(df_dir['movie_title'].unique())))
        logger.info("Average duration of Movie: {}".format(sum(df_dir['duration'])/len(df_dir)))
        logger.info("Languages directed in: {}".format(df_dir['language'].unique()))
        logger.info("Average Budget for a Movie: {} million".format(sum(df_dir['budget'])/(len(df_dir)*100000)))
        logger.info("Average Gross Income for a Movie: {} million".format(sum(df_dir['gross'])/(len(df_dir)*100000)))
        max_budget_row = df_dir.loc[df_dir.index[df_dir['budget'] == max(df_dir['budget'])]] #gives a dataframe for the row with maximum budget value
        logger.info("Highest Budget Movie: \"{}\" with a budget of {} million and gross income of {} million.".format(max_budget_row.iloc[0]['movie_title'][:-1], float(max_budget_row.iloc[0]['budget'])/100000, float(max_budget_row.iloc[0]['gross'])/100000)) #Using [:-1] to remove last unknown character
    