import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as pbar
import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from multisensory_playbook import DetectionTask, DetectionTask_versatile, ClassicalTask
import optuna
from scipy.optimize import fsolve
import re
import time
unisensory_activation = lambda x: x * (x > 0)  # relu: return x if x>0, return 0 if x!>0


def build_input_layer(A, V):
    input_layer = [[] for _ in range(len(A))] # A is a batch 
    for trial in range(len(A)):
        audio_left = unisensory_activation(-A[trial])
        audio_right = unisensory_activation(A[trial])

        video_left = unisensory_activation(-V[trial])
        video_right = unisensory_activation(V[trial])

        input_layer[trial] = torch.from_numpy(
            1 * np.vstack([audio_left, audio_right, video_left, video_right])
        )  # input units x time
    stacked = torch.stack(input_layer, dim=0).permute(2, 0, 1) # timsteps x batchsize x inputdim
    stacked = stacked.to(torch.float32)
    return stacked

def batch_generator_rnn(A, V, y_data, batch_size):
    # A, V are in shape [batch_size, seq_length, features]
    # Convert A and V from numpy array to tensors in order to use CUDA
    A_tensor = torch.from_numpy(A).cuda()  # Convert NumPy array to PyTorch tensor
    V_tensor = torch.from_numpy(V).cuda()  # Convert NumPy array to PyTorch tensor
    
    # Shuffle
    perm = torch.randperm(A_tensor.shape[0]).cuda()
    A_shuffled = A_tensor[perm]
    V_shuffled = V_tensor[perm]
    y_shuffled = y_data[perm]

    # Batches
    n_batches = A_tensor.shape[0] // batch_size

    for i in range(n_batches):
        # Extract batches for A, V, and y_data
        A_batch = A_shuffled[i * batch_size : (i + 1) * batch_size]
        V_batch = V_shuffled[i * batch_size : (i + 1) * batch_size]
        y_batch = y_shuffled[i * batch_size : (i + 1) * batch_size]

        # Check if there is only one class in the batch. If yes, skip this batch.
        """
        if len(torch.unique(y_batch)) == 1:
            print(f"batch {i+1} skipped: singular class in label data, y")
            continue
        """
        A_batch = A_batch.cpu().numpy()
        V_batch = V_batch.cpu().numpy()
        
        #print('y from batchmaker', y_batch.shape)
        yield A_batch, V_batch, y_batch

# initialise V randomly (pweights for prev hidden layer)
def train_rnn(
    model,
    A,
    V,
    y_data,
    batch_size,
    optimizer,
    criterion,
    nb_epochs=10,
    lr=0.001,
    rnn_layer=None,
    linear_layer=None,
    verbose=True
    ):
    
    loss_hist = []
    for epoch in range(nb_epochs):
        batch_loss = []
           
        for A_batch, V_batch, y_batch in batch_generator_rnn(A, V, y_data, batch_size):
                 
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            input_layer = build_input_layer(A_batch, V_batch) # x is 25x4x90 with  (batchsize x inputdim x timestep)         
               
            outputs = model(input_layer, rnn_layer, linear_layer)
            # Compute loss
            loss = criterion(outputs, y_batch)
            batch_loss.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{nb_epochs}], Loss: {loss.item():.4f}')

        
        loss_hist.append(np.mean(batch_loss))
    
    return loss_hist

# Sanity check for network training loss
def check_training_loss_anomalies(epoch_loss):
    # Convert epoch_loss to numpy array if it's not already
    epoch_loss = np.array(epoch_loss)
    
    # Calculate mean and standard deviation
    mean_loss = np.mean(epoch_loss)
    std_loss = np.std(epoch_loss)
    
    # Define anomaly threshold
    threshold = mean_loss + (2 * std_loss)
    
    # Identify anomalies
    anomalies = epoch_loss > threshold
    
    # Print out the anomalies
    if np.any(anomalies):
        anomaly_epochs = np.where(anomalies)[0]
        anomaly_values = epoch_loss[anomalies]
        print(f"Anomalies detected at epochs: {anomaly_epochs}")
        print(f"Loss values: {anomaly_values}")
    else:
        print("No anomalies detected in training loss.")



