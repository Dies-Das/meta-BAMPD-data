import numpy as np
from itertools import product, permutations, combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# Set the figure size for publication
fig_size = (34/3.,27/3.)



T=np.array([4,8,12])#np.arange(4,15)


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


def greedy_q(b):

	Va1 = (b[:,0]+1.)/(b[:,0]+b[:,1]+2.)
	Va2 = (b[:,2]+1.)/(b[:,2]+b[:,3]+2.)
	return  np.absolute(Va1-Va2)


def mean(b):

	Va1 = (b[:,0]+1.)/(b[:,0]+b[:,1]+2.)
	Va2 = (b[:,2]+1.)/(b[:,2]+b[:,3]+2.)
	return Va1,Va2


def variance (b):
	var1 = (b[:,0]+1.)*(b[:,1]+1.)/((b[:,0]+b[:,1]+2.)**2 * (b[:,0]+b[:,1]+3.))
	var2 = (b[:,2]+1.)*(b[:,3]+1.)/((b[:,2]+b[:,3]+2.)**2 * (b[:,2]+b[:,3]+3.))

	return np.sqrt(var1), np.sqrt(var2)


def model_pol (beta,omega,Bels):

	avg = mean(Bels)
	var = variance(Bels)
	
	P1 = np.exp(beta*avg[0] + omega*var[0])
	P2 = np.exp(beta*avg[1] + omega*var[1])
	
	PS = (P1+P2).copy()
	return P1/PS,P2/PS
	


def get_tuple(length, total):
	return list(filter(lambda x:sum(x)==total,product(range(total+1),repeat=length)))
def generate_best_fit():
    TT = 20
    Beliefs = []
    for i in range (TT+1):
        Beliefs.append(np.asarray(get_tuple(4,i)))
    Beliefs = np.vstack(Beliefs)




    num = 20
    BETA = np.linspace (0,20,5*num)
    OMEG = np.linspace (-10,10,5*num)
    dist = np.zeros ((len(BETA),len(OMEG)))

    opt_omeg_c_T = []

    for idx, i in enumerate(T):

        A = np.load(f'Data/all_polls_{i}.npy')
        B = np.load(f'Data/distinct_P{i}.npy')
        C = np.load(f'Data/comp_cost_{i}.npy')/10**-2
        B = np.array(B,dtype=int)
        
        GP = A[-1]
        OP = A[0]

        belly = Beliefs[:len(GP)]
        opt_omeg = np.zeros (len(A[:,0]))
        for j in range (len(A[:,0])):
            
            for k in range (len(BETA)):
                for l in range (len(OMEG)):
                    
                    mp  = model_pol (BETA[k],OMEG[l],belly)[1]

                    
                    bp  = A[j]


                    dist[k,l] = np.std(bp-mp)
                #print (mp.max(),mp.min())
            opt_omeg[j] = OMEG[np.where(dist==dist.min())[1][0]]
        
        CC = np.zeros (len(C))    
        for j in range (len(C)):
            CC[j] = opt_omeg[B[j]]
        
        opt_omeg_c_T.append(CC)
        print (i)
        plt.plot(C,opt_omeg_c_T[-1], label=f'T={i}', linewidth=2, linestyle=line_styles[idx % len(line_styles)],color = 'black')

    #plt.plot(opt_omeg)

    
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('$c$')
    plt.ylabel (r'$\omega$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/omega_fit.png', dpi=300)  # Save with high dpi for clarity
    plt.show()




