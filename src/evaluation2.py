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
exp_name = "evaluation2"
log_file_name = "Log_" + exp_name + ".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
# fh.setLevel(logging.INFO)
logger.addHandler(fh)

from model import Model2
from analysis import dataset_analysis
from preprocess import dataset_preprocess
from util import dataframe_manipulation, tensor_manipulation


import warnings
warnings.filterwarnings('ignore')



def run_fit_genre(train_dataloader, test_dataloader, model, criterion, optimizer, batch_size = 16, epoch = 200):
    for epoch in range(epoch):  # loop over the dataset multiple times
        pred_genre = torch.empty(0)
        act_genre = torch.empty(0)
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels_genre = data #labels and inputs are tensors
            #append to act tensor
            act_genre = torch.cat((act_genre, labels_genre))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output_genre = model(inputs)
            # with torch.no_grad():
            topk = tensor_manipulation.topk_genres(output_genre, k=6) #convert this pre-softmax value to a multi-hot encoded vector
            output_genre=topk.reshape(batch_size,-1) #Since we create new tensors this can lead to computational graph getting cutoff. So we do it in no_grad()
            loss_genre = criterion(output_genre, labels_genre)
            loss_genre.backward()
            optimizer.step()
            #save the predicted labels in the pred tensor
            pred_genre = torch.cat((pred_genre, output_genre))

        pred_genre_test = torch.empty(0)
        act_genre_test = torch.empty(0)
        model.eval()
        with torch.no_grad():
            for i, data_test in enumerate(test_dataloader, 0):
                inputs_test, labels_genre_test = data_test #labels and inputs are tensors
                act_genre_test = torch.cat((act_genre_test, labels_genre_test))
                output_genre_test = model(inputs_test)
                topk = tensor_manipulation.topk_genres(output_genre_test, k=6) #convert this pre-softmax value to a multi-hot encoded vector
                output_genre_test=topk.reshape(1,-1) #Since we create new tensors this can lead to computational graph getting cutoff. So we do it in no_grad()
                #save the predicted labels in the pred tensor
                pred_genre_test = torch.cat((pred_genre_test, output_genre_test))

        #Once this epoch is over, find the accuracy over the pred and act tensors
        metric_genre = Accuracy(task='multiclass', num_classes=output_genre.shape[1])
        print("Epoch {}: Train Accuracy of Genre : = {:.3f} and Test Accuracy of Genre: = {:.3f}".format(epoch+1, metric_genre(pred_genre, act_genre).item()*100, metric_genre(pred_genre_test, act_genre_test).item()*100))
        logger.info("Epoch {}: Train Accuracy of Genre : = {:.3f} and Test Accuracy of Genre: = {:.3f}".format(epoch+1, metric_genre(pred_genre, act_genre).item()*100, metric_genre(pred_genre_test, act_genre_test).item()*100))
    
    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../saved_models", "model_genre.pt")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(model_save_path) # Save




def run_fit_year(train_dataloader, test_dataloader, model, criterion, optimizer, batch_size = 16, epoch = 200):
    for epoch in range(epoch):  # loop over the dataset multiple times
        pred_year = torch.empty(0)
        act_year = torch.empty(0)
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels_year = data #labels and inputs are tensors
            #append to act tensor
            act_year = torch.cat((act_year, labels_year))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output_year = model(inputs)
            loss_year = criterion(output_year, labels_year)
            loss_year.backward()
            optimizer.step()

            #save the predicted labels in the pred tensor
            pred_year = torch.cat((pred_year, output_year))

        #Once this epoch is over, find the accuracy over the pred and act tensors
        metric_year = MeanSquaredError()
        pred_year_test = torch.empty(0)
        act_year_test = torch.empty(0)
        model.eval()
        with torch.no_grad():
            for i, data_test in enumerate(test_dataloader, 0):
                inputs_test, labels_year_test = data_test #labels and inputs are tensors
                act_year_test = torch.cat((act_year_test, labels_year_test))
                output_year_test = model(inputs_test)
                #save the predicted labels in the pred tensor
                pred_year_test = torch.cat((pred_year_test, output_year_test))
        # print("Accuracy of Year for epoch {}: = {:.3f}".format(epoch+1, metric_year(pred_year, act_year).item()*100))
        print("Epoch {}: Training Loss = {:.3f}, (Predicted: {}, Actual: {}), Testing Error = {:.3f} , (Predicted: {}, Actual: {})".format(epoch+1, metric_year(pred_year, act_year).item()*100, pred_year.tolist()[0][0], act_year.tolist()[0][0], metric_year(pred_year_test, act_year_test).item()*100,pred_year_test.tolist()[0][0], act_year_test.tolist()[0][0]))
        logger.info("Epoch {}: Training Loss = {:.3f}, (Predicted: {}, Actual: {}), Testing Error = {:.3f} , (Predicted: {}, Actual: {})".format(epoch+1, metric_year(pred_year, act_year).item()*100, pred_year.tolist()[0][0], act_year.tolist()[0][0], metric_year(pred_year_test, act_year_test).item()*100,pred_year_test.tolist()[0][0], act_year_test.tolist()[0][0]))

    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../saved_models", "model_year.pt")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(model_save_path) # Save


def prepare_dataloaders(df, genre, train_split):
    x_train, y_train, x_test, y_test = dataframe_manipulation.split_test_train(df, genre, train_split)
    dict_of_column_types_input = dataset_preprocess.column_type_dict_input()

    #do label encoding in order to convert into tensor: encoder_train/test will help to get back the original values
    x_train, encoder_train = dataset_preprocess.label_encode(df=x_train, column_type=dict_of_column_types_input)
    x_test, encoder_test = dataset_preprocess.label_encode(df=x_test, column_type=dict_of_column_types_input)

    x_train = (x_train-x_train.mean())/x_train.std()
    x_test = (x_test-x_test.mean())/x_test.std()

    #convert to tensors
    x_train_tensor = torch.from_numpy(x_train.values.astype(np.float32)) #dataframe.values gives a numpy array
    x_test_tensor = torch.from_numpy(x_test.values.astype(np.float32))
    #https://stackoverflow.com/questions/29763620/how-to-select-all-columns-except-one-in-pandas
    y_train_tensor_genre = torch.from_numpy(y_train.loc[:, y_train.columns != 'title_year'].values.astype(np.float32))
    y_test_tensor_genre = torch.from_numpy(y_test.loc[:, y_test.columns != 'title_year'].values.astype(np.float32))
    y_train_tensor_year = torch.from_numpy(y_train.loc[:, y_train.columns == 'title_year'].values.astype(np.float32))
    y_test_tensor_year = torch.from_numpy(y_test.loc[:, y_test.columns == 'title_year'].values.astype(np.float32))

    #create tensor datasets
    train_dataset_genre = TensorDataset(x_train_tensor, y_train_tensor_genre)
    test_dataset_genre = TensorDataset(x_test_tensor, y_test_tensor_genre)
    train_dataset_year = TensorDataset(x_train_tensor, y_train_tensor_year)
    test_dataset_year = TensorDataset(x_test_tensor, y_test_tensor_year)

    #load dataloaders
    train_dataloader_genre = torch.utils.data.DataLoader(train_dataset_genre, batch_size=_batch_size)
    test_dataloader_genre = torch.utils.data.DataLoader(test_dataset_genre, batch_size=1)
    train_dataloader_year = torch.utils.data.DataLoader(train_dataset_year, batch_size=_batch_size)
    test_dataloader_year = torch.utils.data.DataLoader(test_dataset_year, batch_size=1)

    return train_dataloader_genre, test_dataloader_genre, train_dataloader_year, test_dataloader_year, len(x_train.columns)





def prepare_model(input_feature_length, number_of_genres, number_of_hidden_layers, hidden_layer_nodes, _learning_rate):
    _hidden_layers_data = [] #used to dynamically generate the hidden layers
    for i in range(number_of_hidden_layers):
         _hidden_layers_data.append((hidden_layer_nodes, nn.ReLU())) #all hidden layers have ReLU activation function

    model_genre = Model2.Model2Genre(_in_features = input_feature_length, _out_features = number_of_genres, hidden_layers_data = _hidden_layers_data, _learning_rate = _learning_rate,)
    model_year = Model2.Model2Year(_in_features = input_feature_length, hidden_layers_data = _hidden_layers_data, _learning_rate = _learning_rate,)
    criterion_genre = nn.KLDivLoss(reduction="batchmean")
    criterion_year = nn.MSELoss()
    optimizer_genre = optim.Adam(model_genre.parameters(), lr=_learning_rate)
    optimizer_year = optim.Adam(model_year.parameters(), lr=_learning_rate)

    return model_genre, model_year, criterion_genre, criterion_year, optimizer_genre, optimizer_year




def run(_learning_rate=0.01, _batch_size=100, _epoch=100, number_of_hidden_layers=10, hidden_layer_nodes=64):
    dataset_name = "p1_movie_metadata.csv"
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name)
    df = pd.read_csv(dataset_path)
    df = dataset_preprocess.drop_extra_columns(df=df) #Remove worthless columns that won't contribute to result
    df = dataset_preprocess.remove_nan(df=df) #Drop NaN values before proceeding with statistics
    logger.info(df.head().to_markdown())
    # dataset_analysis.run(df=df)
    df, genre = dataset_preprocess.expand_genre(df=df)
    logger.info(df.head().to_markdown())
    train_split = 0.8

    #prepare dataloaders
    train_dataloader_genre, test_dataloader_genre, train_dataloader_year, test_dataloader_year, input_feature_length = prepare_dataloaders(df, genre, train_split)

    #prepare model
    number_of_genres = len(genre.columns)
    model_genre, model_year, criterion_genre, criterion_year, optimizer_genre, optimizer_year = prepare_model(input_feature_length, number_of_genres, number_of_hidden_layers, hidden_layer_nodes, _learning_rate)

    #run fit
    run_fit_genre(train_dataloader_genre, test_dataloader_genre, model_genre, criterion_genre, optimizer_genre, _batch_size, _epoch)
    run_fit_year(train_dataloader_year, test_dataloader_year, model_year, criterion_year, optimizer_year, _batch_size, _epoch)


if __name__=="__main__":
    _learning_rate = 0.1
    _batch_size = 10
    _epoch = 10
    _number_of_hidden_layers=4
    _hidden_layer_nodes=20
    run(_learning_rate, _batch_size, _epoch, _number_of_hidden_layers, _hidden_layer_nodes)