
'''
Computing the average time at which exploratory action is done under a given environment.
'''
import numpy as np
from itertools import product, permutations, combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from multiprocessing import Pool
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
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


def zeroth_rows_of_powers(A, N):
    current_row = A[0, :]  # Start with the 0th row of A
    result = [current_row]  # Store the initial row (A^1)

    for _ in range(1, N):
        current_row = np.dot(current_row, A)
        result.append(current_row)  # Append the result after each multiplication

    return np.array(result) 


def pol_trans (pol,trans,p1,p2,time):
    TT = trans.copy()
    for i in range (len(trans)):
        if (len(np.where (trans[i] == 1.21)[0])==0) and (len(np.where (trans[i] == 1.11)[0])==0):
            break
        next_win2 = np.where (trans[i] == 1.21)[0][0]
        next_los2 = np.where (trans[i] == 1.20)[0][0]
        next_win1 = np.where (trans[i] == 1.11)[0][0]
        next_los1 = np.where (trans[i] == 1.10)[0][0]

        act = pol[i]
        
        if (act == 0):
            TT[i][next_win2] = 0
            TT[i][next_los2] = 0
            TT[i][next_win1] = p1
            TT[i][next_los1] = 1. - p1
        elif (act == 1):
            TT[i][next_win1] = 0
            TT[i][next_los1] = 0
            TT[i][next_win2] = p2
            TT[i][next_los2] = 1. - p2
        elif (act == 0.5):
            TT[i][next_win2] = 0.5*p2
            TT[i][next_los2] = 0.5*(1. - p2)
            TT[i][next_win1] = 0.5*p1
            TT[i][next_los1] = 0.5*(1. - p1)


 
    T_pow  =    zeroth_rows_of_powers(TT, time-1)     
    return TT,T_pow






def get_tuple(length, total):
	return list(filter(lambda x:sum(x)==total,product(range(total+1),repeat=length)))
	
def flatten(xss):
    return [x for xs in xss for x in xs]
TTs = 20
Beliefs = []
for i in range (TTs+1):
	Beliefs.append(np.asarray(get_tuple(4,i)))
Beliefss = np.vstack(Beliefs)

##Generating strictly exploratory policy
VA1,VA2 = mean(Beliefss)
pollie = np.heaviside(VA1-VA2,0.5)
EPP = np.zeros (len(Beliefss))
for i in range (len(EPP)):
	if (VA1[i] !=VA2[i]):
		EPP[i] = pollie[i]
	else:
		n1 = Beliefss[i][0]+Beliefss[i][1]
		n2 = Beliefss[i][2]+Beliefss[i][3]

		EPP[i] = np.heaviside (n1-n2,0.5)
		





#EP is exploratory/computatory policy
def process_row(j, A, Trans, P1, P2, EP, Beliefss, i):
    avg_t_local = np.zeros((len(P1), len(P2)))
    print(j, len(A[:,0]))
    for l in range(len(P1)):
        for m in range(l, len(P2)):
            p1 = P1[l]
            p2 = P2[m]
            TT, T_pow = pol_trans(A[j], Trans, p1, p2, i)
            avt = 0
            inddy = np.where(np.absolute(A[j]-EP)<=0.5)[0]	#For exploratory
            #inddy = np.where(A[j]!=EP)[0]	#For computatory

            for k in inddy:
              if (A[j][k] == 0.5):
                avt += Beliefss[k].sum() * T_pow[Beliefss[k].sum() - 1][k]*A[j][k]
              else:
                avt += Beliefss[k].sum() * T_pow[Beliefss[k].sum() - 1][k]

            avg_t_local[l, m] = avt
            avg_t_local[m, l] = avt
    return j, avg_t_local



def generate_plots_comp():

    for idx, i in enumerate(T):

        A = np.load(f'Data/all_polls_{i}.npy')
        GP = A[-1]
        OP = A[0]
        EP = EPP.copy()[:len(GP)]
        
        
        Trans = np.zeros ((len(flatten(Beliefs[:i+1])),len(flatten(Beliefs[:i+1]))))
        for k in range (len(GP)):
            bi = flatten(Beliefs)[k]
            K = np.tile(bi, (1, 4,1))
            identity_matrix = np.eye(4) 
            K = K+ identity_matrix

            for j in range (4):
                indy = np.where((flatten(Beliefs) == K[0][j]).all(axis=1))[0]
                veccy = np.array([1.11, 1.1, 1.21, 1.2])
                Trans [k,indy] = veccy[j]
        
        
        
        
        P1 = np.linspace (0,1,50)
        P2 = P1.copy()
        print ('now we go')
        avg_t = np.zeros ((len(A[:,0]),len(P1),len(P2)))
        
        def callback(result):
            j, local_result = result
            avg_t[j] = local_result
            print(f"Completed processing row {j}")

        belly = Beliefs[:len(GP)]
        with Pool(processes = 8) as pool:
            for j in range (len(A[:,0])):
                pool.apply_async(process_row, args=(j, A, Trans, P1, P2, EP, Beliefss, i), callback=callback)
            pool.close()
            pool.join()
            np.save('Data/avg_t_mat_{}'.format(i),avg_t)
    

    '''    
    '''

    #TODO: line226 may not have the best measure. figure it out!
    #A gives avg tau matrix, for each pair of reward probs and number of distinct policies
    for idx, i in enumerate(T):
        A = np.load(f'Data/avg_t_mat_{i}.npy') / i
        B = np.load(f'Data/distinct_P{i}.npy')
        C = np.load(f'Data/comp_cost_{i}.npy')
        B = np.array(B, dtype=int)
        
        print(A.shape)
        CC = np.zeros((len(C), A.shape[1], A.shape[2]))

        dc = C[1] - C[0]
        for j in range(len(C)):
            CC[j] = A[B[j]].copy()
        
        Var_C = np.zeros(A[0].shape)
        print(Var_C.shape)
        for j in range(Var_C.shape[1]):
            for k in range(Var_C.shape[0]):
                Var_C[j, k] = np.square(np.diff(CC[:, j, k]) / dc).mean()
        
        fig, ax = plt.subplots()
        cax = ax.imshow(Var_C, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        ax.set_xlabel('$p_1$')
        ax.set_ylabel(r'$p_2$')
        
        # Format axes with scientific notation
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_label(r'$\chi_\tau$')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
        plt.tight_layout()
        plt.savefig(f'plots/tau_susc_{i}.png', dpi=300)  # Save with high dpi for clarity
