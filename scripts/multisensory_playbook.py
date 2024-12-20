from numpy.random import choice, rand, randn
import numpy as np
import lea  # probability calculations, see https://pypi.org/project/lea/
from collections import defaultdict
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import copy
import contourpy as cp
import pandas as pd
import random
import numba
from numba import jit, njit
from numba import types
from numba.typed import Dict
import joblib
import math


def plog(p):
    if p == 0:
        return -1e100  # don't set to -inf because it makes some stuff nan
    else:
        return np.log(p)
# Change boolean input to explicit 0,1 input

# Expand the detection params search to include handling of sliding window features
def detection_params_search_test(p_ranges, nb_trials, nb_steps, random_seed, tasktype='DetectionTask', trans_prob=None, time_dep=0, k=1, windowsize_list=[3]):
    # Tested and passed: seed passed in via processes is received correctly
    
    # Sample task parameters
    # set unique seeds to each processor
    np.random.seed(random_seed)
    
    while True:
        p = {
            k: rand() * (upper - lower) + lower
            for k, (lower, upper) in p_ranges.items()
        }
        # Add debugging prints for parameter values
        #print("Sampled parameters:", p)

        if p["pc"] <= 0.5 * p["pn"]:
            continue
        if p["pi"] >= p["pc"]:
            continue
        if p["pi"] >= 0.5 * p["pn"]:
            continue
        if p["pi"] + p["pc"] <= p["pn"]:
            continue
        if p["pc"] + p["pi"] > 1.0:
            continue
        break
    
    # Generate trials
    # For testing: p = {'pm': 0.058777852497873013, 'pe': 0.9181276195517423, 'pc': 0.43995704240058564, 'pn': 0.05267577898779354, 'pi': 0.09027316303027239}
    if tasktype == 'DetectionTask':
        """
        Generate separate task and trials from those tasks for train and test.
        Reason: __post_init__ within class DetectionTask implements a random seed, if provided. 
        Its good practice to control for this even if you don't pass in the seed (for future mishaps)
        They must have different seeds
        """

        task = DetectionTask(**p)
        full_trials_train = task.generate_trials(nb_trials, nb_steps)
        full_trials_test = task.generate_trials(nb_trials, nb_steps)

    elif tasktype == 'DetectionTask_versatile': 
        task = DetectionTask_versatile(**p)#, random_seed=random_seed_train) 
        full_trials_train = task.generate_trials(nb_trials, nb_steps, time_dep, k) # uses the seed provided above
        full_trials_test = task.generate_trials(nb_trials, nb_steps, time_dep, k)
        print('Train-Test generated') # logging

    # Generate test data separately
    training_size = nb_trials
    testing_size = nb_trials
    training_trials = Trials(
        repeats=training_size,
        time_steps=nb_steps,
        M=full_trials_train.M,
        A=full_trials_train.A,
        V=full_trials_train.V,
        task=task
    )

    testing_trials = Trials(
        repeats=testing_size,
        time_steps=nb_steps,
        M=full_trials_test.M,
        A=full_trials_test.A,
        V=full_trials_test.V,
        task=task
    )
    
    # Calculate accuracy
    accs_tmp = []

    for pairs in [0, 1, 2]:
        
        # Check if there is only one class in training data. If yes, skip 
        unique_classes_train = np.unique(training_trials.M)
        unique_classes_test = np.unique(testing_trials.M)
        if len(unique_classes_train) == 1:
            #print(f"Skipping training and testing: Only one class ({unique_classes_train[0]}) in training data.")
            #print(f"Params: {p}")
            return [0.0, 0.0, 0.0], np.array(list(p.values())) * 0.0 # function exits here
            
        if len(unique_classes_test) == 1:
            #print(f"Skipping training and testing: Only one class ({unique_classes_test[0]}) in test data.")
            #print(f"Params: {p}")
            return [0.0, 0.0, 0.0], np.array(list(p.values())) * 0.0
        if pairs in [0,1]:
            classifier = LinearClassifier(task, pairs=pairs)
  
            # Train and test the classifier using trials generated using sliding window features
            trained_classifier = classifier.train(training_trials)
            res = trained_classifier.test(testing_trials)
            accs_tmp.append(res.accuracy)
        if pairs == 2:
            for ws in windowsize_list:
                classifier = LinearClassifier(task, pairs=pairs, windowsize=ws)
    
                # Train and test the classifier using trials generated using sliding window features
                trained_classifier = classifier.train(training_trials)
                res = trained_classifier.test(testing_trials)
                accs_tmp.append(res.accuracy)
                
    # Filter for accuracy
    _, a = np.unique(full_trials_train.M, return_counts=True)  # majority class classifier
    a = a.max() / a.sum()
    w = 1 - a
    c = (1 + a) / 2
     
    if (max(accs_tmp) > (c - w / 2 * 0.8)) & (min(accs_tmp) < (c + w / 2 * 0.8)): # 0.75
        return accs_tmp, np.array(list(p.values()))
    else:
        #print('filtered')
        return [0.0] * np.man(windowsize_list), np.array(list(p.values())) * 0.0 
    

@dataclass
class Task:
    @property
    def random_variables(self):
        return NotImplemented

    def generate_trials(self, repeats, time_steps, random_seed=None):
        #print(random_seed)
        """
        Generates trials for Detection Task
        Other tasks have their own functions for trial generation within the respective Task classes
        """
        if random_seed is not None:
            np.random.seed(random_seed)  # Set the random seed if provided
        # random variables
        rv = self.random_variables
        M = rv["M"]
        A = rv["A"]
        V = rv["V"]
        #E = rv["E"]
        # cache calculated joint distribution
        joint_dists = {}
        for m in [-1, 0, 1]:
            if lea.P(M == m) == 0:
                continue
            joint_dists[m] = lea.joint(A, V).given(M == m).calc()
        # generate true target values
        arr_M = np.array(M.random(repeats))
        
        steps = np.array(
            [joint_dists[m].random(time_steps) for m in arr_M]
        )  # steps has shape (repeats, time_steps, 2)
        if time_steps == 0:
            # print(steps.shape)
            return Trials(
                repeats=repeats,
                time_steps=time_steps,
                task=self,
                M=arr_M,
                A=steps[:, None],
                V=steps[:, None],
            )
        else:
            return Trials(
                repeats=repeats,
                time_steps=time_steps,
                task=self,
                M=arr_M,
                A=steps[:, :, 0],
                V=steps[:, :, 1],
               
            )

    @property
    def baseline(self):
        if not hasattr(self, "_baseline"):
            M = self.random_variables["M"]
            self._baseline = max([lea.P(M == m) for m in [-1, 0, 1]])
        return self._baseline

    def baseline_reward(self, reward):
        M = self.random_variables["M"]
        probs = np.array([lea.P(M == m) for m in [-1, 0, 1]])
        expected_rewards = np.einsum("m,mg->g", probs, reward)
        return np.max(expected_rewards)


@dataclass
class DetectionTask(Task):
    pm: float
    pe: float
    pn: float
    pc: float
    pi: float
    random_seed: int = None
    """
    Generate trials from the function inside Task class
    """
    @property
    def random_variables(self, random_seed=None):
        """
        if random_seed is not None: # __post_init__ should take care of this, and this is just a redundant implementation of random seed
            random.seed(random_seed)  # Set the random seed if provided
        """
        target = lea.pmf({-1: self.pm * 0.5, 1: self.pm * 0.5, 0: 1 - self.pm})
        emit_if_target = lea.event(self.pe)
        emit_if_no_target = lea.event(0.0)
        emit = target.switch(
            {-1: emit_if_target, 1: emit_if_target, 0: emit_if_no_target}
        )
        signal_dist = {
            (-1, True): lea.pmf({-1: self.pc, +1: self.pi, 0: 1 - self.pc - self.pi}),
            (+1, True): lea.pmf({+1: self.pc, -1: self.pi, 0: 1 - self.pc - self.pi}),
            (0, True): lea.pmf({-1: 0, +1: 0, 0: 1.0}),  # cannot happen
            (-1, False): lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
            (0, False): lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
            (+1, False): lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
        }
        signal = lea.joint(target, emit).switch(signal_dist)
        signal_A, signal_V = signal.clone(n=2, shared=(target, emit))
        self._random_vars = {"M": target, "E": emit, "A": signal_A, "V": signal_V}
        return self._random_vars



@dataclass
class DetectionTask_versatile(Task): # can generate time dependent and time-independent detection tasks
    pm: float
    pe: float
    pn: float
    pc: float
    pi: float
    trans_prob: dict = None
    time_dep: int = 0
    k: int = 1


    
    random_seed: int = None  # Added a field for random seed
    do_return: bool = True

    """
            # Time independent task
            self.trans_prob = {
                0: [1-self.pe, self.pe],  # Probabilities for transitioning from E=0
                1: [1-self.pe, self.pe],  # Probabilities for transitioning from E=1
            }
    """
    
    def generate_trials(self, nb_trials, nb_steps): 
        # Extract local ON-duration, k and time_dep
        k = self.k
        time_dep = self.time_dep
        
        # Initialize numpy arrays for M, A, V and E
        arr_M = choice([-1, 0, 1], size=nb_trials, p=[self.pm / 2, 1 - self.pm, self.pm / 2]) # pm is 1, therefore it is ok to leave the 0 in
        arr_A = np.zeros((nb_trials, nb_steps-k), dtype=int) 
        arr_V = np.zeros((nb_trials, nb_steps-k), dtype=int)
        arr_E = np.zeros((nb_trials, nb_steps-k), dtype=int)

        
    
        for trial in range(nb_trials):
            M = arr_M[trial]
            #attempt = 1
            #base_e = choice([1, 0], size=nb_steps, p=[self.pe, 1-self.pe])               # Choose E for the current trial                
            while True:
                while True:  # Loop until a valid base_e is generated
                    base_e = choice([1, 0], size=nb_steps, p=[self.pe, 1-self.pe])  # Choose E for the current trial
                    
                    if np.sum(base_e) != 0:  # Exit the loop if there is at least one 1 in base_e
                        #attempt+=1
                        #print(attempt)
                        break
                        
                
                idx = np.where(base_e==1)[0]                                                    # Indices where Et = 1
                fin_e = base_e.copy()                                                        # Start with a copy. When k=1, E = base_e = fin_e  
                # Modify indices as per the value of k
                for i in range(1, k):
                    idx_plus = idx + i  # Compute next indices
                    idx_plus = idx_plus[idx_plus < nb_steps]  # Keep only indices within bounds
                    if idx_plus.size == 0:
                        continue
                    fin_e[idx_plus] = 1  # Set the calculated indices to 1
                # Remove buffer
                E = fin_e[k:]
       
                assert len(E) == nb_steps-k
                if np.sum(E) != 0:  # Exit the loop if there is at least one 1 in base_e
                        break
                


            e0 = np.array([-1, 0, 1]) # Add noise if E = 0
            p_e0 = np.array([self.pn / 2, 1 - self.pn, self.pn / 2])
            e1 = np.array([-M, 0, M]) # add probabilities for incorrectness
            p_e1 = np.array([self.pi, 1 + (- self.pc - self.pi), self.pc])

            A = np.where(E, choice(e1, size=E.size, p=p_e1), choice(e0, size=E.size, p=p_e0))
            V = np.where(E, choice(e1, size=E.size, p=p_e1), choice(e0, size=E.size, p=p_e0))
            
            arr_A[trial, :] = A 
            arr_V[trial, :] = V
            arr_E[trial, :] = E
        
        return Trials(repeats=nb_trials, time_steps=nb_steps, M=arr_M, A=arr_A, V=arr_V, E=arr_E, task=self)


    """
    # Function to generate emission variables for subsequent time steps based on Markovian nature
    @property
    def random_variables(self, random_seed=None):
        if hasattr(self, "_random_vars"): 
            return self._random_vars
        
        # Generate M similar to DetectionTask
        target = lea.pmf({-1: self.pm * 0.5, 1: self.pm * 0.5, 0: 1 - self.pm})

        # Generate the initial emission probability
        emit_initial = lea.pmf({1: self.pe, 0: 1 - self.pe})

        # Create a chain of emission probabilities based on trans_prob
        emits = [emit_initial]
        for _ in range(1, self.nb_steps):
            prev_emit = emits[-1]
            # Calculate next emission probabilities based on trans_prob
            next_emit_0 = prev_emit.p(0) * self.trans_prob[0][0] + prev_emit.p(1) * self.trans_prob[1][0]
            next_emit_1 = prev_emit.p(0) * self.trans_prob[0][1] + prev_emit.p(1) * self.trans_prob[1][1]
            emits.append(lea.pmf({0: next_emit_0, 1: next_emit_1}))

        # Generate signal variables for A and V considering the Markovian nature of emission
        signals_A = []
        signals_V = []
        for emit in emits:
            signal_A = emit.switch({
                0: lea.pmf({-1: self.pn * 0.5, 1: self.pn * 0.5, 0: 1 - self.pn}),
                1: target.switch({
                    -1: lea.pmf({-1: self.pc, 1: self.pi, 0: 1 - self.pc - self.pi}),
                    1: lea.pmf({1: self.pc, -1: self.pi, 0: 1 - self.pc - self.pi}),
                    0: lea.pmf({-1: 0, 1: 0, 0: 1.0})  # Cannot happen
                })
            })
            signal_V = signal_A.clone()  # Assuming V follows the same distribution as A
            signals_A.append(signal_A)
            signals_V.append(signal_V)

        self._random_vars = {'M': target, 'A': signals_A[-1], 'V': signals_V[-1]}
        return self._random_vars
    """


@dataclass
class ClassicalTask(Task):
    s: float

    @property
    def pc(self):
        return (1 + 2 * self.s) / 3

    @property
    def pi(self):
        return (1 - self.s) / 3

    pn = pi

    @property
    def random_variables(self):
        if not hasattr(self, "_random_vars"):
            M = lea.pmf({-1: 0.5, 1: 0.5})
            S = M.switch(
                {
                    -1: lea.pmf({-1: self.pc, +1: self.pi, 0: self.pn}),
                    +1: lea.pmf({+1: self.pc, -1: self.pi, 0: self.pn}),
                }
            )
            A, V = S.clone(n=2, shared=(M,))
            self._random_vars = dict(M=M, A=A, V=V)
        return self._random_vars


@dataclass
class Trials:
    repeats: int
    time_steps: int
    M: np.ndarray
    A: np.ndarray
    V: np.ndarray
    task: Task
    E: np.ndarray = field(default_factory=lambda: np.array([]))

    """
    Has been re-implemented to incorporate window sliding
    
    """
    
    # Re-implementation of counts to incorporate sliding window functionality 
    def counts(self, windowsize=None, pairs=2): 
        A = self.A
        V = self.V
        print('win size: ', windowsize)

        def calculate_state(draw_sequence):
            # Mapping for the states to digits
            state_to_digit = {-1: 0, 0: 1, 1: 2} # To change -1 to 1
            
            # Convert the draw sequence to a base-3 number
            base_3_number = 0
            for draw in draw_sequence:
                base_3_number = base_3_number * 3 + state_to_digit[draw]
            
            # The state is the base-3 number
            return base_3_number

        def apply_state(row):
            # Convert row to list and pass it to the calculate_state function
            return calculate_state(row.tolist())
        
        if self.time_steps == 0:
            return np.zeros((self.repeats, 6 + 3 * pairs))
        else:
            if pairs == 0:
                CA = np.apply_along_axis(np.bincount, 1, A + 1, minlength=3)  # (repeats, 3)
                CV = np.apply_along_axis(np.bincount, 1, V + 1, minlength=3)  # (repeats, 3)
                C = np.concatenate((CA, CV), axis=1)

            elif pairs == 1:
                AV = (A + 1) + 3 * (V + 1)  # shape (repeats, time_steps)
                C = np.apply_along_axis(np.bincount, 1, AV, minlength=9)  # (repeats, 9)     

            elif pairs == 2: # consider windows with n number of consecutive AV-pairs
                max_state = 3**(2*windowsize) # 3**(2n) 
                C = np.zeros((self.repeats, max_state))
                for trialnum in range(self.repeats):
                    _A = A[trialnum]
                    _V = V[trialnum]
                    df = pd.DataFrame()
                    df['A'], df['V'] = _A, _V
                    
                    if windowsize == 2:
                        df['A-1'], df['V-1'] =  df['A'].shift(1), df['V'].shift(1) # Shifting column down one step
                    if windowsize == 3:
                        df['A-1'], df['V-1'] =  df['A'].shift(1), df['V'].shift(1) # Shifting column down one step
                        df['A-2'], df['V-2'] =  df['A'].shift(2), df['V'].shift(2) # Shifting column down one step (window size is 3)
                    if windowsize == 4:
                        df['A-1'], df['V-1'] =  df['A'].shift(1), df['V'].shift(1) # Shifting column down one step
                        df['A-2'], df['V-2'] =  df['A'].shift(2), df['V'].shift(2) # Shifting column down one step (window size is 3)
                        df['A-3'], df['V-3'] =  df['A'].shift(3), df['V'].shift(3) # Shifting column down one step (window size is 4)
                
                    df = df.dropna()
                    
                    # Apply the function to each row and store the result in a new column 'state'
                    df['state'] = df.apply(apply_state, axis=1) # make values positive (less confusing)
                    # Calculate value counts
                    state_counts = df['state'].value_counts()
                    

                    # Generate a range of numbers representing all possible states
                    # Adjust the range based on your specific needs (max_state + 1)
                    all_possible_states = range(0, max_state)  # Replace max_state with your actual maximum state value

                    # Reindex the value counts to include all possible states
                    # Fill missing values (states with 0 occurrences) with 0
                    state_counts = state_counts.reindex(all_possible_states, fill_value=0)
                    C[trialnum,:] = state_counts
                
        return C

    _singleton_labels = ["A=-1", "A=0", "A=1", "V=-1", "V=0", "V=1"]
    _coincidence_labels = [f"({a}, {v})" for v in [-1, 0, 1] for a in [-1, 0, 1]]

    def count_labels(self, pairs=1):
        if pairs:
            return self._coincidence_labels
        else:
            return self._singleton_labels

    def simple_joint_counts(self, pairs=1):
        Cs = self.counts(pairs=False).T
        C = [
            Cs[0] + Cs[3],  # net count in favour of -1
            Cs[1] + Cs[4],  # net count in favour of 0
            Cs[2] + Cs[5],  # net count in favour of +1
        ]
        if pairs:
            Cc = dict(
                (k, v)
                for k, v in zip(self._coincidence_labels, self.counts(pairs=1).T)
            )
            C = C + [Cc["(-1, -1)"], Cc["(1, 1)"]]
        return np.array(C).T

    joint_count_labels = ["-1", "0", "1", "(-1, -1)", "(1, 1)"]


class Classifier:
    def __init__(self, task, pairs=True, reward=None, windowsize=None):
        self.task = task
        self.pairs = pairs
        self.reward = reward
        self.windowsize=windowsize
        self.basename = f"{self.__class__.__name__}"

    @property
    def name(self):
        n = self.basename
        if self.pairs:
            n += "/pairs"
        else:
            n += "/singletons"
        if self.reward is not None:
            n += "/reward"
        return n

    def train(self, trials):
        return self  # default no training needed

    def test(self, trials):
        return NotImplemented


class LinearClassifier(Classifier):
    def __init__(
        self, task, pairs=True, reward=None, windowsize=None, model=linear_model.LogisticRegression
    ):
        super().__init__(task, pairs=pairs, reward=reward, windowsize=windowsize)
        self.model_class = model
        self.basename = model.__name__
        #self.pipeline = None ### TEST

    def _vector(self, trials):
        return trials.counts(pairs=self.pairs, windowsize=self.windowsize)

    def train(self, trials, max_iter=120000):
        classifier = copy.copy(self)
        classifier.training_trials = trials
        classifier.model = self.model_class(max_iter=max_iter)
        counts = self._vector(trials)
        # print(counts)
        classifier.pipeline = make_pipeline(StandardScaler(), classifier.model)
        classifier.pipeline.fit(counts, trials.M)

        # Collect training metrics
        train_pred = classifier.pipeline.predict(counts)
        precision, recall, f1, support = precision_recall_fscore_support(trials.M, train_pred, average=None, labels=[-1, 1], zero_division=0) # since we have only two classes,  macro averaging isn't necessary: class structure is symmetric      
        #print(trials.M)
        class_distribution = np.array([np.sum(trials.M == -1), np.sum(trials.M == 1)])
        #print(class_distribution)
        classifier.train_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'class_distribution': class_distribution
        }
        
        
        return classifier # unlike test, no need to invoke Results for the metrics

    def test(self, trials):
        counts = self._vector(trials)
        pred = self.pipeline.predict(counts)
        precision, recall, f1, support = precision_recall_fscore_support(trials.M, pred, average=None, labels=[-1, 1], zero_division=0) # since we have only two classes,  macro averaging isn't necessary: class structure is symmetric      
        class_distribution = np.array([np.sum(trials.M == -1), np.sum(trials.M == 1)])
        #test_metrics = self._calculate_metrics(trials.M, pred)                                                                       
        
        return Results(
            trials=trials,
            predictions=pred,
            classifier=self,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            class_distribution=class_distribution
        )
        
    def _calculate_metrics(self, true, pred):
        precision, recall, f1, support = precision_recall_fscore_support(
            true, pred, average=None, labels=[-1, 1], zero_division=0
        )
        class_distribution = np.bincount(true + 1)  # Shift -1 to 0 for proper indexing
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'class_distribution': class_distribution
        }
        """
        return Results(
            trials=trials,
            predictions=pred,
            classifier=self,
        )
        """
    def get_train_metrics(self):
        if hasattr(self, 'train_metrics'):
            return self.train_metrics
        else:
            raise AttributeError("Training metrics have not been calculated yet. Make sure to train the classifier first.")
        
    def save_model(self, filepath):
        joblib.dump(self.pipeline, filepath)
    
    def load_model(self, filepath):
        self.pipeline = joblib.load(filepath)

@dataclass
class Results:
    trials: Trials
    predictions: np.ndarray
    classifier: Classifier
    evidence: Optional[np.ndarray] = None
    precision: Optional[np.ndarray] = None
    recall: Optional[np.ndarray] = None
    f1: Optional[np.ndarray] = None
    support: Optional[np.ndarray] = None
    class_distribution: Optional[np.ndarray] = None
   

    @property
    def accuracy(self):
        if not hasattr(self, "_accuracy"):
            self._accuracy = (
                sum(self.predictions == self.trials.M) / self.trials.repeats
            )
            n = 3 # significant digits
            raw_accuracy = (sum(self.predictions == self.trials.M) / self.trials.repeats)
        
            scale = 10 ** (n - 1 - int(math.floor(math.log10(abs(raw_accuracy)))))
            self._accuracy = round(raw_accuracy * scale) / scale

        return self._accuracy
    
    @property
    def metrics_summary(self):
        return pd.DataFrame({
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
            'Support': self.support,
            'class_distribution': self.class_distribution
            
        }, index=[-1, 1])

    """
    @property
    def class_distribution(self):
        total = np.sum(self.support)
        return pd.DataFrame({
            'Count': self.support,
            'Percentage': self.support / total * 100
        }, index=[-1, 1])
    """
    @property
    def confusions(self):
        # TODO: not tested
        if not hasattr(self, "_confusions"):
            C = self._confusions = np.zeros((3, 3), dtype=int)
            np.add.at(C, (self.trials.M + 1, self.predictions + 1), 1)
        return self._confusions

    @property
    def confusions_normalised(self):
        C = self.confusions
        C = (C * 1.0) / np.sum(C, axis=1)[:, None]
        return C

    def compute_expected_reward(self, reward):
        R = reward  # indices are (M, G)
        G = self.predictions  # indices are (repeat,)
        M = self.trials.M  # indices are (repeat,)
        return R[M + 1, G + 1].sum() / len(M)

    @property
    def expected_reward(self):
        if not hasattr(self, "_expected_reward"):
            self._expected_reward = self.compute_expected_reward(
                reward=self.classifier.reward
            )
        return self._expected_reward


