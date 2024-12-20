import sys
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
from multisensory_playbook import DetectionTask, DetectionTask_versatile, ClassicalTask, Trials
from RNN_helpers import build_input_layer, batch_generator_rnn, train_rnn, check_training_loss_anomalies

from scipy.optimize import fsolve
import re
import time
from pylab import *
from scipy.optimize import fsolve
from numpy.random import binomial, random_sample


# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! GPU support is enabled.")
else:
    print("CUDA is not available. Please check your installation.")

# Set the default tensor type to CUDA tensors if CUDA is available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_device('cuda')

# Number of GPUs available
num_gpus = torch.cuda.device_count()
#print(f"Number of GPUs available: {num_gpus}")

# Details of each GPU
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
    print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")


def set_global_seed(seed):
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    np.random.seed(seed)
    #print(f"Global seed set to: {seed}")
def calculate_pg(ff, k, N=90, correction=1):
    """
    Use ff to calulate pg, given a k and N
    ff: desired filtered fraction of E 
    pg: probability of E(t)=1 in the base_e (generator)
    k : local on-time duration
    N : number time-steps
    """
    buffer = k
    #pg = (1-fsolve(lambda x: ff-(1-x**k)/(1-x**(N)), 0.9))[0] 
    if correction:
        ff = (1-fsolve(lambda x: ff-(1-x**k)/(1-x**(N+int(buffer)-1*(k-1))), 0.9))[0]

    return ff
def levy_dist(lmax):
    l = arange(lmax+1) # allow l=0 but must have p=0
    pl = zeros(lmax+1)
    pl[1:] = 1.0/l[1:]**2
    pl[:] /= sum(pl)
    return pl
    
def generate_mix_samples(pg, pl, N=90, repeats=10):
    np.random.seed(seed)
    lmax = len(pl)-1
    #print('lmax ', lmax)
    #pgl = pg*pl
    M = N+lmax-1
    all_E = []
    positions = arange(M)
    lengths = arange(lmax+1)
    for _ in range(repeats):
        keep_going = True
        while keep_going:
            # generate nonzero points
            num_nonzero = binomial(M, pg)
            if num_nonzero==0:
                continue
            E_starts = choice(positions, size=num_nonzero, replace=False)
            L = choice(lengths, size=num_nonzero, p=pl)
            #print('L ', L)
            E = zeros(M, dtype=bool)
            for e_start, l in zip(E_starts, L):
                E[e_start:e_start+l] = 1
            E = E[lmax-1:]
            assert len(E)==N
            keep_going = (sum(E)==0)
        all_E.append(E)
    all_E = array(all_E) # shape (repeats, N)
    return all_E

def estimate_fraction_on(pg, pl, N=90, repeats=1000):
    return generate_mix_samples(pg, pl, N=N, repeats=repeats).mean()

def generate_levy_AV(pm, pn, pi, pc, nb_trials, nb_steps, E):
    arr_M = choice([-1, 0, 1], size=nb_trials, p=[pm / 2, 1 - pm, pm / 2])
    arr_A = np.zeros((nb_trials, nb_steps), dtype=int) # Q1! + k padding in the begining of E for Levy flights?
    arr_V = np.zeros((nb_trials, nb_steps), dtype=int)
    arr_E = E #np.zeros((nb_trials, nb_steps-k), dtype=int)

    for trial in range(nb_trials):
        M = arr_M[trial]
        e0 = np.array([-1, 0, 1]) # Add noise if E = 0
        p_e0 = np.array([pn / 2, 1 - pn, pn / 2])
        e1 = np.array([-M, 0, M]) # add probabilities for incorrectness
        p_e1 = np.array([pi, 1 + (- pc - pi), pc])
        
        A = np.where(E[trial], choice(e1, size=E[trial].size, p=p_e1), choice(e0, size=E[trial].size, p=p_e0))
        V = np.where(E[trial], choice(e1, size=E[trial].size, p=p_e1), choice(e0, size=E[trial].size, p=p_e0))
        arr_A[trial, :] = A 
        arr_V[trial, :] = V
        #arr_E[trial, :] = E[trial]
            
    return arr_M, arr_A, arr_V,arr_E 
seed = int(sys.argv[1])
#seeds = [1000, 2000, 3000, 4000, 5000]
#for seed in seeds:
print(seed)
#nb_steps = 500 
path = f"./data/RNN/RNN_train_levy_seed_{seed}_{nb_steps}steps_eqldistr_k"
# Activation functions
unisensory_activation = lambda x: x * (x > 0)  # relu: return x if x>0, return 0 if x!>0
set_global_seed(seed)
    
# Network 
nb_inputs = 4
nb_hidden = 100 # units in hidden layer 150
nb_layers = 1 # hidden layers
nb_outputs = 3
nb_epochs = 900
repeats = 1 # number of networks trained, original = 5
batch_size = 32 # Each batch has batch_size items
positive_weights = False
nb_trials = batch_size * 500 # original: 10000
learning_rate = 1e-6
pe_sparse = 0.04

# Detection task, new versatile formulation
time_dep = 1 # 1: there is time dependence

#pc = 0.45
#pm=1
    


lmax = 8 #levy k max
pl = 8 * [1/8] # equal distribution
pm, pn, pi, pc, nb_steps = 1, 1/3, 0.01, 0.45, 500
E = generate_mix_samples(pg=pe_sparse, pl=pl, N=nb_steps, repeats=nb_trials)

tasks = [
    
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+1-1, k=1), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=1), # sparse    
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+2-1,k=2), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=2), # sparse  
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+3-1,k=3), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=3), # sparse
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+4-1,k=4), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=4), # sparse
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+5-1,k=5), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=5), # sparse 
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+6-1,k=6), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=6), # sparse 
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+7-1,k=7), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=7), # sparse
    DetectionTask_versatile(pm=pm, pe=calculate_pg(pe_sparse,  N=nb_steps+8-1,k=8), pc=pc, pn=1 / 3, pi=0.01, time_dep=time_dep, k=8) # sparse
    
]
task_keys = ['Sparse Levy']

# Generate training data 
training_size = nb_trials
M, A, V, E = generate_levy_AV(pm, pn, pi, pc, nb_trials, nb_steps, E)
full_trials_train = Trials(
    repeats=training_size,
    time_steps=nb_steps+lmax-1,
    M=M,
    A=A,
    V=V,
    task=None
)

full_trials_test_list = [task.generate_trials(nb_trials, nb_steps+kk+1-1) for kk,task in enumerate(tasks)]



# Generate labels
y_train, A_train, V_train = torch.tensor(full_trials_train.M + 1), full_trials_train.A, full_trials_train.V
test_acc_dict = {}

    
# Create a model instance
rnn_layer = nn.RNN(input_size=nb_inputs, hidden_size=nb_hidden, nonlinearity='relu', batch_first=False)

linear_layer = nn.Linear(nb_hidden, nb_outputs)
       
def model(input_data, rnn_layer, linear_layer):
    input_data = input_data.cuda()
    batchsize = input_data.shape[1]
    h0 = torch.zeros((1, batchsize, nb_hidden), dtype=torch.float32).cuda()
    out, h1 = rnn_layer(input_data, h0)
    lin_out = linear_layer(out)
    sum_out = lin_out.sum(axis=0).cuda()
    return sum_out
                          

optimizer = torch.optim.Adam(list(rnn_layer.parameters()) + list(linear_layer.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss in PyTorch internally applies a LogSoftmax layer to its inputs before computing the negative log likelihood loss. 

epoch_loss = train_rnn( # loss from each epoch
        model,
    A_train,
    V_train,
    y_train,
    batch_size,
    optimizer=optimizer,
    criterion=criterion,
    nb_epochs=nb_epochs,
    lr=learning_rate,
    rnn_layer=rnn_layer,
    linear_layer=linear_layer,
    verbose=1)
print('Training complete')
# Check anomalies in training loss
check_training_loss_anomalies(epoch_loss)
del full_trials_train, y_train, A_train, V_train

# Save the model checkpoint
# Save the model checkpoint
torch.save({
    'rnn_layer_state_dict': rnn_layer.state_dict(),
    'linear_layer_state_dict': linear_layer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, path + f'_checkpoint.pkl')


for kk, full_trials_test in enumerate(full_trials_test_list):
    test_k = kk+1
    y_test, A_test, V_test = torch.tensor(full_trials_test.M + 1), full_trials_test.A, full_trials_test.V

    # Load the model checkpoint
    checkpoint = torch.load(path + f'_checkpoint.pkl')
    rnn_layer.load_state_dict(checkpoint['rnn_layer_state_dict'])
    linear_layer.load_state_dict(checkpoint['linear_layer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set the model to evaluation mode
    rnn_layer.eval()
    linear_layer.eval()
    with torch.no_grad():
        input_layer = build_input_layer(A_test, V_test)
        y_pred = model(input_layer, rnn_layer, linear_layer).cpu().numpy()
        y_pred_argmax = np.argmax(y_pred, axis=1)
        test_acc = (np.mean(y_pred_argmax == y_test.cpu().numpy()))*100
        test_acc_dict[('levy', test_k)] = test_acc
    
    # Clear memory after each task
    del full_trials_test, y_test, A_test, V_test, input_layer, y_pred, y_pred_argmax
    torch.cuda.empty_cache()

# Save the dictionary to a file
with open(path+'_testaccuracies.pkl', 'wb') as file:
    pickle.dump(test_acc_dict, file)
