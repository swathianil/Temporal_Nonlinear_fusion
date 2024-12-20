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
import sys, os
path = "./data"

pairs = int(sys.argv[1])

train = 1
print(train)
# "s_range": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Sparse to Dense, sparse is less than 0.2
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
    
# Tasks
# Detection task
pe_sparse = 0.04
nb_steps = 500 
nb_trials = 100000 # Original: 100000
classifier_type = LinearClassifier
time_dep = 1 # 1: there is time dependence

pm=1
pc = 0.45



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
  #windowsize_list = [None]
    windowsize_list = prefix_list # Rename windowsize for saving in dictionary
print(f'windowsize_list: {windowsize_list}')    
modelpath = './data/'

# Training all classifiers
if train:
    for a, task in enumerate(tasks):
        train_k = k_list[a]
        print(task)
    
        full_trials_train = task.generate_trials(nb_trials, nb_steps+train_k-1)
    
        # Generate training data 
        training_size = nb_trials
        training_trials = Trials(
            repeats=training_size,
            time_steps=nb_steps+train_k-1,
            M=full_trials_train.M,
            A=full_trials_train.A,
            V=full_trials_train.V,
            task=task
        )
    
        #pairs = 2
        
        for windowsize in windowsize_list:
            
            classifier = LinearClassifier(task, pairs=pairs, windowsize=windowsize)
    
            # Train the classifier using trials generated using sliding window features
            trained_classifier = classifier.train(training_trials)
            
        
            trained_classifiers_dict[(train_k, windowsize)] = trained_classifier
            train_metrics = trained_classifier.get_train_metrics()
            print(f"k: {train_k},  w: {windowsize}")
            train_metrics_dict[(train_k, windowsize)] = train_metrics
    # Save the dictionary to a file
    """
    full_path = f'/trained_classifiers_{nb_steps}steps.pkl'
    # Check if file exists and is not empty
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
      # File exists and is not empty, so load the dictionary
        with open(full_path, 'rb') as file:
          loaded_trained_classifiers_dict = pickle.load(file)
        print("Loaded existing trained classifiers from file.")
        loaded_trained_classifiers_dict.update(trained_classifiers_dict) # Add current trained classifiers
      # Write the updated dictionary back to the file
        with open(full_path, 'wb') as file:
            pickle.dump(loaded_trained_classifiers_dict, file)
        print("Updated and saved trained classifiers to file.")
        
        
    else:
    """
    if pairs in [0,1]:
        with open(f'./{prefix_list[pairs]}_classifier_trainmetrics_500steps.pkl', 'wb') as file:
              pickle.dump(pickle.dump(train_metrics_dict, file), file)
    else:
        with open('./NLFw_classifier_trainmetrics_500steps.pkl', 'wb') as file:
          pickle.dump(train_metrics_dict, file)
        
    if pairs in [0,1]:
        with open(f'./{prefix_list[pairs]}_trained_classifiers_{nb_steps}steps.pkl', 'wb') as file:
            pickle.dump(trained_classifiers_dict, file)
    else:
        with open(f'./NLFw_trained_classifiers_{nb_steps}steps.pkl', 'wb') as file:
            pickle.dump(trained_classifiers_dict, file)
        
    print("Created new file and saved trained classifiers.")    

else:
    file_path = f'./data/{prefix_list[pairs]}_trainedclassifier/{prefix_list[pairs]}_trained_classifiers_{nb_steps}steps.pkl'

    with open(file_path, 'rb') as file:
        trained_classifiers_dict = pickle.load(file)
    
acc_dict = {}
metrics_dict = {}
for train_k in k_list:
    for windowsize in windowsize_list:
        trained_classifier = trained_classifiers_dict[(train_k, windowsize)]
        for a, task in enumerate(tasks):
            test_k = k_list[a]
            # Generate test data 
            testing_size = nb_trials
            full_trials_test = task.generate_trials(nb_trials, nb_steps+test_k-1)
            
            testing_trials = Trials(
                repeats=testing_size,
                time_steps=nb_steps+train_k-1,
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
            acc_dict[(train_k,windowsize,test_k)] = accs
            metrics_dict[(train_k, windowsize, test_k)] = {
                'accuracy': res.accuracy,
                'precision': res.precision.tolist(),  # Convert numpy array to list for easier storage
                'recall': res.recall.tolist(),
                'f1': res.f1.tolist(),
                'support': res.support.tolist(),
                'class_distribution': res.class_distribution.tolist()
            }

# Save the dictionary to a file
"""
full_path = f'/classifier_accuracies{nb_steps}steps.pkl'
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
#with open(f'./{prefix_list[pairs]}_classifier_accuracies_500steps.pkl', 'wb') as file:
#  pickle.dump(pickle.dump(acc_dict, file), file)
print('debug1')
if pairs in [0,1]:
    with open(f'./{prefix_list[pairs]}_classifier_testmetrics_500steps.pkl', 'wb') as file:
      pickle.dump(pickle.dump(metrics_dict, file), file)
else:
    with open('./NLFw_classifier_testmetrics_500steps.pkl', 'wb') as file:
      pickle.dump(metrics_dict, file)
