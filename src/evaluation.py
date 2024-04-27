import pandas as pd
import numpy as np
# import matplotplib as plt
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
from torchmetrics import Accuracy
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


# #Based on https://stackoverflow.com/questions/58984043/how-to-retrieve-topk-values-across-multiple-rows-along-with-their-respective-ind
# def topk_genres(output_genres, k=4, prob_sum_thresold=0.95):
#     result_tensor = torch.empty(0)
#     for row in output_genres: #We have to do it element-wise. Using flatten() will consider topk over all m*n elements
#         row_tensor = []
#         _, linear_indices = row.topk(k)
#         topk_indices = linear_indices % row.shape[-1]
#         sum=0
#         for i, val in enumerate(row):
#             if i in topk_indices and sum<=prob_sum_thresold: #We consider upto k top elements or till probability sum exceeds a threshold, whichever comes earlier
#                 row_tensor.append(1) #If in the topk of that row, add as possible genre
#                 sum+=val.data
#             else:
#                 row_tensor.append(0) #if not in the topk, not a possible genre
#         result_tensor = torch.cat((result_tensor, torch.as_tensor(row_tensor)))
#     return result_tensor


#Based on https://stackoverflow.com/questions/58984043/how-to-retrieve-topk-values-across-multiple-rows-along-with-their-respective-ind
def topk_genres(output_genres, k=4, prob_sum_thresold=0.95):
    for row in output_genres: #We have to do it element-wise. Using flatten() will consider topk over all m*n elements
        _, linear_indices = row.topk(k)
        topk_indices = linear_indices % row.shape[-1]
        sum=0
        for i, val in enumerate(row):
            if i in topk_indices and sum<=prob_sum_thresold: #We consider upto k top elements or till probability sum exceeds a threshold, whichever comes earlier
                sum+=val.data
                with torch.no_grad():
                    row[i]=torch.tensor(1) #If in the topk of that row, add as possible genre
            else:
                with torch.no_grad():
                    row[i]=torch.tensor(0) #if not in the topk, not a possible genre
    return output_genres



def run_fit(train_dataloader, model, criterion, optimizer, batch_size = 16, epoch = 200):
        for epoch in range(epoch):  # loop over the dataset multiple times
            pred_genre = torch.empty(0)
            act_genre = torch.empty(0)
            pred_year = torch.empty(0)
            act_year = torch.empty(0)
            running_loss = 0.0
            
            #Break the tuple into two
            criterion_genre = criterion[0] 
            criterion_year = criterion[1]

            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data #labels and inputs are tensors
                labels_genre, labels_year = labels[:, 0:-1], labels[:, -1].reshape(-1,1) #Split the labels based on genre and year: genre is n*m and year is n*1 tensor
                #append to act tensor
                act_genre = torch.cat((act_genre, labels_genre))
                act_year = torch.cat((act_year, labels_year))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output_genre, output_year = model(inputs)
                with torch.no_grad():
                    topk = topk_genres(output_genre, k=6) #convert this pre-softmax value to a multi-hot encoded vector
                    output_genre=topk.reshape(1,-1) #Since we create new tensors this can lead to computational graph getting cutoff. So we do it in no_grad()
                
                logger.info("Predicted Genre = {}\nActual Genre = {}".format(output_genre, labels_genre))
                logger.info("Predicted Year = {}\nActual Year = {}".format(output_year, labels_year))
                loss_genre = criterion_genre(output_genre, labels_genre)
                loss_year = criterion_year(output_year, labels_year)

                loss = loss_genre + loss_year
                logger.info("Total loss = {}".format(loss))

                # # loss_genre.requires_grad=True
                # # loss_year.requires_grad=True
                # output_genre.requires_grad=True
                # output_year.requires_grad=True

                # loss_genre.retain_grad()
                # loss_year.retain_grad()
                # output_genre.retain_grad()
                # output_year.retain_grad()

                # loss.backward()

                # logger.info("loss_genre.grad = {}".format(loss_genre.grad))
                # logger.info("loss_year.grad = {}".format(loss_year.grad))
                # logger.info("output_genre.grad = {}".format(output_genre.grad))
                # logger.info("output_year.grad = {}".format(output_year.grad))
                

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % batch_size == batch_size-1:    # print every 2000 mini-batches
                    logger.info('[{:d}, {:d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / batch_size))
                    running_loss = 0.0
                #save the predicted labels in the pred tensor
                pred_genre = torch.cat((pred_genre, output_genre))
                pred_year = torch.cat((pred_year, output_year))

            #Once this epoch is over, find the accuracy over the pred and act tensors
            metric_genre = Accuracy(task='multiclass', num_classes=output_genre.shape[1])
            metric_year = MeanSquaredError()
            logger.info("Accuracy of Genre for epoch {}: = {:.3f}".format(epoch+1, metric_genre(pred_genre, act_genre).item()*100))
            logger.info("Accuracy of Year for epoch {}: = {:.3f}".format(epoch+1, metric_year(pred_year, act_year).item()*100))
            logger.info("Predicted Genre = {}\nActual Genre = {}".format(pred_genre, act_genre))
            logger.info("Predicted Year = {}\nActual Year = {}".format(pred_year, act_year))
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

    #do label encoding in order to convert into tensor: encoder_train/test will help to get back the original values
    x_train, encoder_train = dataset_preprocess.label_encode(df=x_train, column_type=dict_of_column_types_input)
    x_test, encoder_test = dataset_preprocess.label_encode(df=x_test, column_type=dict_of_column_types_input)

    # logger.info("\n-------------------------------------x_train after encoding-------------------------------------")
    # logger.info(x_train.head().to_markdown())
    # logger.info("\n-------------------------------------x_test after encoding-------------------------------------")
    # logger.info(x_test.head().to_markdown())

    #convert to tensors
    x_train_tensor = torch.from_numpy(x_train.values.astype(np.float32)) #dataframe.values gives a numpy array
    y_train_tensor = torch.from_numpy(y_train.values.astype(np.float32))
    x_test_tensor = torch.from_numpy(x_test.values.astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.values.astype(np.float32))

    # logger.info("\n-------------------------------------x_train tensor-------------------------------------")
    # logger.info(x_train_tensor)
    # logger.info("\n-------------------------------------y_train tensor-------------------------------------")
    # logger.info(y_train_tensor)
    # logger.info("\n-------------------------------------x_test tensor-------------------------------------")
    # logger.info(x_test_tensor)
    # logger.info("\n-------------------------------------y_test tensor-------------------------------------")
    # logger.info(y_test_tensor)

    #create tensor datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # logger.info("\n-------------------------------------train_dataset-------------------------------------")
    # logger.info(train_dataset)
    # logger.info("\n-------------------------------------test_dataset-------------------------------------")
    # logger.info(test_dataset)

    #load dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size)

    #model details
    model = Model1.Model1(_in_features = len(x_train.columns), _out_features = len(genre.columns)+1, _learning_rate = _learning_rate,)
    criterion = (nn.CrossEntropyLoss(), nn.MSELoss()) #[0]:For one-hot encoded loss (genres), [1]:For regression loss (year)
    optimizer = optim.Adam(model.parameters())

    #run fit
    run_fit(train_dataloader, model, criterion, optimizer, _batch_size, _epoch)


if __name__=="__main__":
    _learning_rate = 0.01
    _batch_size = 1
    _epoch = 100
    run(_learning_rate, _batch_size, _epoch)