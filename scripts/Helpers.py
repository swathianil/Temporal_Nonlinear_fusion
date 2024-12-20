import numpy as np
from scipy.optimize import fsolve
from pylab import *
import matplotlib.pyplot as plt

# Plotting standards
def plotting_style():
    # Set global parameters
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.color'] = 'red'
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['xtick.labelsize'] = 18

    plt.rcParams['ytick.labelsize'] = 18

# Calculate fraction of E=1 in generator sequence
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

# Explore shapes of datastructures
def print_structure(obj, level=0):
    indent = "  " * level
    if isinstance(obj, tuple):
        print(f"{indent}Tuple: Length {len(obj)}")
        for item in obj:
            print_structure(item, level + 1)
    elif isinstance(obj, list):
        print(f"{indent}List: Length {len(obj)}")
        for item in obj:
            print_structure(item, level + 1)
    else:
        print(f"{indent}Type: {type(obj)}")
#print_tuple_structure(test)


def plot_trial_data(A, V, M, E=0, llim=0, rlim=100):
    Mlabel = {0: '-', 1: '>', -1: '<'}

    time = np.arange(0, rlim - llim, 1)
    plt.figure(figsize=(3, 1.5))

    plt.scatter(time[A == -1], np.zeros_like(time[A == -1]), marker='<', s=50, c='purple')
    plt.scatter(time[A == 0], np.zeros_like(time[A == 0]), marker=' ', s=50)
    plt.scatter(time[A == 1], np.zeros_like(time[A == 1]), marker='>', s=50, c='red')

    plt.scatter(time[V == -1], np.ones_like(time[V == -1]) * 0.3, marker='<', s=50, c='purple')
    plt.scatter(time[V == 0], np.ones_like(time[V == 0]) * 0.3, marker=' ', s=50)
    plt.scatter(time[V == 1], np.ones_like(time[V == 1]) * 0.3, marker='>', s=50, c='red')

    if np.mean(E) != 0: # vertical lines if E is provided
        E = E.reset_index(drop=True)
        locations = np.where(E == 1)[0]
        for location in locations:
            if location == locations[-1]:
                plt.axvline(time[location], 0, 1, color='orchid', linestyle='solid', alpha=0.25, label="E = 1")
            else:
                plt.axvline(time[location], 0, 1, color='orchid', linestyle='solid', alpha=0.25)

    plt.legend(frameon=False)
    plt.title(f"M: {M}")
    plt.xticks(time)
    plt.ylim(top=1., bottom=-1.)
    plt.yticks([])
    plt.xticks(np.arange(time[0], time[-1], 5))

    # Remove the box enclosing the plot
    ax = plt.gca()  # Get the current axes instance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel('Steps')


# Plot time series and loglog plots
def timeseriesplot(x, rlim, y, label=None, log=0):
    if log:
        plt.loglog(x[:rlim], y, marker='|', ms = 5, linewidth = 1, label=label)
        plt.title('Log–log plot')
        
    else:
        plt.plot(x[:rlim], y, marker='|', ms = 5, linewidth = 1, label=label)
        ax = plt.gca()  # Get the current axis
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))  # This will force scientific notation for all numbers
        ax.xaxis.set_major_formatter(formatter)
        
    plt.ylabel('Time (s)')
    plt.xlabel('Trials')
    plt.legend(frameon=False)
    plt.legend(frameon=False)
    

## Percentage of A = V = M
def AV_equalitytest(df):
    matching_rows = df[(df['A'] == df['M']) & (df['V'] == df['M'])]
    # Count occurrences per trial
    counts_per_trial = matching_rows.groupby('Trial').size().values
    df_examine = pd.DataFrame()
    df_examine['strength'] = range_s
    df_examine['A,V = M (%)'] = np.round((counts_per_trial/n) * 100, 2)
    return df_examine
    
## Proportion of A, V
def AV_proportiontest(df):
    # Create a pivot table summarizing the counts for 'A'
    summary_A = np.round((df.pivot_table(index='Trial', columns='A', aggfunc='size', fill_value=0) / n * 100),1)
    summary_A.columns = ['A_' + str(int(col)) for col in summary_A.columns]

    # Create a pivot table summarizing the counts for 'V'
    summary_V = np.round((df.pivot_table(index='Trial', columns='V', aggfunc='size', fill_value=0) / n * 100),1)
    summary_V.columns = ['V_' + str(int(col)) for col in summary_V.columns]

    # Merge the two summaries
    summary_df = pd.merge(summary_A, summary_V, left_index=True, right_index=True)
    summary_df['strength'] = range_s
    #summary_df = summary_df.drop(columns=['Trial'])
    return summary_df

## Levy flight helpers
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