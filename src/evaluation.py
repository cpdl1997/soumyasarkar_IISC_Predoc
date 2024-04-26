import pandas as pd
import numpy as np
# import matplotplib as plt
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError

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


import warnings
warnings.filterwarnings('ignore')


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



def run_fit(train_dataloader, model, criterion, optimizer, batch_size = 16, epoch = 200):
        for epoch in range(epoch):  # loop over the dataset multiple times
            pred = torch.empty(0)
            act = torch.empty(0)
            running_loss = 0.0
            
            criterion_genre = criterion[0]
            criterion_year = criterion[1]

            for i, data in enumerate(train_dataloader, 0):
                # get the inputs
                inputs, labels = data
                print(type(inputs), " AND ",  type(labels))
                labels_genre, labels_year = labels[0:-1], labels[0:-1]
                #save the output labels in the act tensor
                act = torch.cat((act, labels))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = torch.reshape(model(inputs), (1,-1))[0]
                output_genre, output_year = output
                
                loss_genre = criterion_genre(output_genre, labels_genre)
                loss_year = criterion_year(output, labels_year)

                loss = loss_genre + loss_year

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % batch_size == batch_size-1:    # print every 2000 mini-batches
                    logger.debug('[{:d}, {:d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / batch_size))
                    running_loss = 0.0
                #save the predicted labels in the pred tensor
                pred = torch.cat((pred, output))

            #Once this epoch is over, find the accuracy over the pred and act tensors
            metric = MulticlassAccuracy()
            logger.debug("Accuracy for epoch {}: = {:.3f}".format(epoch+1, metric(pred, act).item()*100))
            # print("Accuracy for this epoch: %.3f"%(metric(pred, act).item()*100))





def run(_learning_rate=0.01, _batch_size=100, _epoch=100):
    dataset_name = "p1_movie_metadata.csv"
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name)
    df = pd.read_csv(dataset_path)
    df = dataset_preprocess.drop_extra_columns(df=df) #Remove worthless columns that won't contribute to result
    df = dataset_preprocess.remove_nan(df=df) #Drop NaN values before proceeding with statistics
    logger.info(df.head().to_markdown())
    # dataset_analysis.run(df=df)
    df, genre = dataset_preprocess.expand_genre(df=df)
    train_split = 0.8

    #model details
    model = Model1.Model1(_in_features = len(df.columns)-2, _out_features = len(genre.columns), _learning_rate = _learning_rate,)
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss())
    optimizer = optim.Adam(model.parameters())

    x_train, y_train, x_test, y_test = split_test_train(df, genre, train_split)
    dict_of_column_types_input = dataset_preprocess.column_type_dict_input()

    # logger.info("\n-------------------------------------x_train before encoding-------------------------------------")
    # logger.info(x_train.head().to_markdown())
    # logger.info("\n-------------------------------------y_train before encoding-------------------------------------")
    # logger.info(y_train.head().to_markdown())
    # logger.info("\n-------------------------------------x_test before encoding-------------------------------------")
    # logger.info(x_test.head().to_markdown())
    # logger.info("\n-------------------------------------y_test before encoding-------------------------------------")
    # logger.info(y_test.head().to_markdown())

    #do label encoding in order to convert into tensor
    x_train = dataset_preprocess.label_encode(df=x_train, column_type=dict_of_column_types_input)
    x_test = dataset_preprocess.label_encode(df=x_test, column_type=dict_of_column_types_input)

    # logger.info("\n-------------------------------------x_train after encoding-------------------------------------")
    # logger.info(x_train.head().to_markdown())
    # logger.info("\n-------------------------------------x_test after encoding-------------------------------------")
    # logger.info(x_test.head().to_markdown())

    print(x_train.values)
    #convert to tensors
    x_train_tensor = torch.from_numpy(x_train.values) #dataframe.values gives a numpy array
    x_test_tensor = torch.from_numpy(x_test.values)
    y_train_tensor = torch.from_numpy(y_train.values)
    y_test_tensor = torch.from_numpy(y_test.values)
    #create tensor datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    #load dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = _batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = _batch_size)

    #run fit
    run_fit(train_dataloader, model, criterion, optimizer, _batch_size, _epoch)


if __name__=="__main__":
    _learning_rate = 0.01
    _batch_size = 100
    _epoch = 100
    run(_learning_rate, _batch_size, _epoch)