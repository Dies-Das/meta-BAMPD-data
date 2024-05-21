import numpy as np
from itertools import product, permutations, combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def generate_plots():
    # Set the figure size for publication
    fig_size = (34/3.,27/3.)



    T=np.array([4,8,12])#np.arange(6,12,2)


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # Configure Matplotlib's RC params for font size, tick direction, and tick width
    plt.rcParams['axes.labelsize'] = 28  # Increased label size
    plt.rcParams['xtick.labelsize'] = 18  # Increased x tick label size
    plt.rcParams['ytick.labelsize'] = 18  # Increased y tick label size
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.5  # Making x ticks thicker
    plt.rcParams['ytick.major.width'] = 1.5  # Making y ticks thicker
    plt.rcParams['axes.linewidth'] = 2.5  # Making axes thicker
    plt.rcParams['legend.fontsize'] = 14  # Increase font size for legends
    plt.rcParams['xtick.major.size'] = 7  # Increase x tick length
    plt.rcParams['ytick.major.size'] = 7  # Increase y tick length


    line_styles = ['-', '--', '-.']



    # Plotting total expected rewards
    #plt.figure(figsize=fig_size)
    for idx, i in enumerate(T):
        A = np.load(f'Data/comp_cost_{i}.npy')*10**-2
        B = np.load(f'Data/total_exp_rew_{i}.npy')
        
        B -= B.min()
        B /= B.max()
        
        plt.plot(A, B, label=f'T={i}', linewidth=2, linestyle=line_styles[idx % len(line_styles)],color = 'black')
        
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('$c$')
    plt.ylabel (r'$\frac{V-V^g}{V^*-V^g}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/total_exp_rewards.png', dpi=300)  # Save with high dpi for clarity
    plt.clf()

    # Plotting average computation time
    #plt.figure(figsize=fig_size)
    for idx, i in enumerate(T):
        A = np.load(f'Data/comp_cost_{i}.npy')*10**-2
        B = np.load(f'Data/avg_comp_time_{i}.npy')
        
        B -= B.min()
        B /= i
        
        plt.plot(A, B, label=f'T={i}', linewidth=2, linestyle=line_styles[idx % len(line_styles)],color = 'black')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('$c$')
    plt.ylabel (r'$\frac{\langle \tau_c \rangle}{T}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/avg_comp_time.png', dpi=300)
    plt.clf()
'''
'''
##########################################################################################################################

def greedy_q(b):

	Va1 = (b[:,0]+1.)/(b[:,0]+b[:,1]+2.)
	Va2 = (b[:,2]+1.)/(b[:,2]+b[:,3]+2.)
	return np.absolute(Va1-Va2)

def get_tuple(length, total):
	return list(filter(lambda x:sum(x)==total,product(range(total+1),repeat=length)))

if __name__=='__main__':
      generate_plots()