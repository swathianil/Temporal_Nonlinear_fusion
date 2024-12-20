from multisensory_playbook import (
    ClassicalTask,
    DetectionTask,
    DetectionTask_versatile,
    LinearClassifier,
    Trials
)
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import fsolve
import joblib
from pylab import *
from scipy.optimize import fsolve
from numpy.random import binomial, random_sample
import sys, os
path = "./data"

pairs = int(sys.argv[1])

# "s_range": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Sparse to Dense, sparse is less than 0.2
def levy_solve_for_pg(ff_desired, pl, N=500): 
    cpl=cumsum(pl) 

    lmax = len(pl)-1 
    
    lengths = arange(lmax+1) 
    
    fs = lambda pg: 1-prod(1-pg+pg*cpl) 
    
    ff = lambda pg: fs(pg)/(1-prod((1-pg*pl[1:])**(N+lengths[1:]-1))) 
    
    pg = fsolve(lambda pg: ff_desired-ff(pg), ff_desired)[0] 
    
    return pg 


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
            

# Tasks
# Detection task
pe_sparse = 0.04

classifier_type = LinearClassifier
time_dep = 1 # 1: there is time dependence



lmax = 8 #levy k max
pl = levy_dist(lmax) #8 * [1/8] 
pm, pn, pi, pc, nb_trials, nb_steps = 1, 1/3, 0.01, 0.45, 100000, 500#200
E = generate_mix_samples(pg=levy_solve_for_pg(pe_sparse, pl, N=nb_steps), pl=pl, N=nb_steps, repeats=nb_trials)

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

k_list = [1,2,3,4,5,6,7,8]
ideal_dict = {}
trained_classifiers_dict = {}
train_metrics_dict = {}
prefix_list = ['LF', 'NLF_1']
if pairs == 2:
    windowsize_list = [2,3]
if pairs in [0,1]:
    windowsize_list = prefix_list # Rename windowsize for saving in dictionary
modelpath = './data/'

# Training all classifiers


# Generate training data 
training_size = nb_trials
M, A, V, E = generate_levy_AV(pm, pn, pi, pc, nb_trials, nb_steps, E)
training_trials = Trials(
    repeats=training_size,
    time_steps=nb_steps+lmax-1,
    M=M,
    A=A,
    V=V,
    task=None
)



#pairs = 2

for windowsize in windowsize_list:
    classifier = LinearClassifier(None, pairs=pairs, windowsize=windowsize)
    # Train the classifier using trials generated using sliding window features
    trained_classifier = classifier.train(training_trials)
    trained_classifiers_dict[('levy', windowsize)] = trained_classifier
    train_metrics = trained_classifier.get_train_metrics()
    train_metrics_dict[('levy', windowsize)] = train_metrics
    print(f"w: {windowsize}")

# Save the dictionary to a file
"""
full_path = path+'/trained_classifiers_500steps_Levy_eqldistr_k.pkl'
# Check if file exists and is not empty
if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
  # File exists and is not empty, so load the dictionary
  with open(full_path, 'rb') as file:
      loaded_trained_classifiers_dict = pickle.load(file)
      print("Loaded existing trained classifiers from file.")
      loaded_trained_classifiers_dict.update(trained_classifiers_dict) # Add current trained classifiers
      pickle.dump(loaded_trained_classifiers_dict, file)
else:
"""
if pairs in [0,1]:
    with open(f'./{prefix_list[pairs]}_classifier_trainmetrics_500steps_levy.pkl', 'wb') as file:
          pickle.dump(pickle.dump(train_metrics_dict, file), file)
else:
    with open('./NLFw_classifier_trainmetrics_500steps_levy.pkl', 'wb') as file:
      pickle.dump(train_metrics_dict, file)
if pairs in [0,1]:
        with open(f'./{prefix_list[pairs]}_trained_classifiers_{nb_steps}steps_Levy.pkl', 'wb') as file:
            pickle.dump(trained_classifiers_dict, file)
else:
    with open(f'./NLFw_trained_classifiers_{nb_steps}steps_Levy.pkl', 'wb') as file:
        pickle.dump(trained_classifiers_dict, file)
    
print("Created new file and saved trained classifiers.") 


acc_dict = {}
metrics_dict = {}

for windowsize in windowsize_list:
    trained_classifier = trained_classifiers_dict[('levy', windowsize)]
    for a, task in enumerate(tasks):
        test_k = k_list[a]
        # Generate test data 
        testing_size = nb_trials
        full_trials_test = task.generate_trials(nb_trials, nb_steps+test_k-1)
        
        testing_trials = Trials(
            repeats=testing_size,
            time_steps=nb_steps+lmax-1,
            M=full_trials_test.M,
            A=full_trials_test.A,
            V=full_trials_test.V,
            task=task
        )

        # Calculate accuracy
        #pairs = 2
        classifier = trained_classifier
        res = classifier.test(testing_trials)
        accs = res.accuracy
        if pairs in [0,1]:
           windowsize = prefix_list[pairs] # Rename windowsize for saving in dictionary
        acc_dict[('levy', windowsize, test_k)] = accs
        metrics_dict[('levy', windowsize, test_k)] = {
                'accuracy': res.accuracy,
                'precision': res.precision.tolist(),  # Convert numpy array to list for easier storage
                'recall': res.recall.tolist(),
                'f1': res.f1.tolist(),
                'support': res.support.tolist(),
                'class_distribution': res.class_distribution.tolist()
            }
print(f"pl: {pl}")


# Save the dictionary to a file
"""
full_path = path+'/classifier_accuracies_500steps_levy_eqldistr_k.pkl'
# Check if file exists and is not empty
if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
    # File exists and is not empty, so load the dictionary
    with open(full_path, 'rb') as file:
        loaded_acc_dict = pickle.load(file)
        print("Loaded existing classifier accuracies from file.")
        loaded_acc_dict.update(trained_classifiers_dict) # Add current trained classifiers
        pickle.dump(loaded_acc_dict, file)
else:
"""
if pairs in [0,1]:
    with open(f'./{prefix_list[pairs]}_classifier_testmetrics_500steps_levy.pkl', 'wb') as file:
      pickle.dump(pickle.dump(metrics_dict, file), file)
else:
    with open('./NLFw_classifier_testmetrics_500steps_levy.pkl', 'wb') as file:
      pickle.dump(metrics_dict, file)

