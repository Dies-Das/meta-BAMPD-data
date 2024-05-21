

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
fig_size = (34/3., 27/3.)


T = np.array([4, 8, 12])  # np.arange(6,15)


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


def get_tuple(length, total):
    return list(filter(lambda x: sum(x) == total, product(range(total+1), repeat=length)))


def flatten(xss):
    return [x for xs in xss for x in xs]


TTs = 20
Beliefs = []
for i in range(TTs+1):
    Beliefs.append(np.asarray(get_tuple(4, i)))
Beliefss = np.vstack(Beliefs)


def val_pol(pol, i, T, trans, p1, p2):
    # print (i,T,i+T)
    state = Beliefss[i]
    t = state.sum()

    if (t == T):
        return 0

    else:
        act = pol[i]
        if (act == 1):
            next_win = np.where(trans[i] == 1.21)[0][0]
            next_los = np.where(trans[i] == 1.20)[0][0]

            p_win = p2
            return p_win * (1. + val_pol(pol, next_win, T, trans, p1, p2)) + (1.-p_win)*val_pol(pol, next_los, T, trans, p1, p2)
        elif (act == 0):
            next_win = np.where(trans[i] == 1.11)[0][0]
            next_los = np.where(trans[i] == 1.10)[0][0]

            p_win = p1
            return p_win * (1. + val_pol(pol, next_win, T, trans, p1, p2)) + (1.-p_win)*val_pol(pol, next_los, T, trans, p1, p2)
        elif (act == 0.5):
            next_win2 = np.where(trans[i] == 1.21)[0][0]
            next_los2 = np.where(trans[i] == 1.20)[0][0]
            next_win1 = np.where(trans[i] == 1.11)[0][0]
            next_los1 = np.where(trans[i] == 1.10)[0][0]

            p_win2 = p2
            p_win1 = p1
            val1 = p_win1 * (1. + val_pol(pol, next_win1, T, trans, p1, p2)) + \
                (1.-p_win1)*val_pol(pol, next_los1, T, trans, p1, p2)
            val2 = p_win2 * (1. + val_pol(pol, next_win2, T, trans, p1, p2)) + \
                (1.-p_win2)*val_pol(pol, next_los2, T, trans, p1, p2)
            return 0.5*val1+0.5*val2

        else:
            return 'BC'


def process_row(j, A, Trans, P1, P2, Beliefss, i):
    avg_t_local = np.zeros((len(P1), len(P2)))
    print(j, len(A[:, 0]))
    for l in range(len(P1)):
        for m in range(l, len(P2)):
            p1 = P1[l]
            p2 = P2[m]

            avt = val_pol(A[j], 0, i, Trans, p1, p2)

            avg_t_local[l, m] = avt
            avg_t_local[m, l] = avt
    return j, avg_t_local



def generate_plots_individual():
    for idx, i in enumerate(T):

        A = np.load(f'Data/all_polls_{i}.npy')
        GP = A[-1]
        OP = A[0]

        Trans = np.zeros(
            (len(flatten(Beliefs[:i+1])), len(flatten(Beliefs[:i+1]))))
        for k in range(len(GP)):
            bi = flatten(Beliefs)[k]
            K = np.tile(bi, (1, 4, 1))
            identity_matrix = np.eye(4)
            K = K + identity_matrix

            for j in range(4):
                indy = np.where((flatten(Beliefs) == K[0][j]).all(axis=1))[0]
                veccy = np.array([1.11, 1.1, 1.21, 1.2])
                Trans[k, indy] = veccy[j]

        P1 = np.linspace(0, 1, 50)
        P2 = P1.copy()
        print('now we go')
        avg_t = np.zeros((len(A[:, 0]), len(P1), len(P2)))
        def callback(result):
            j, local_result = result
            avg_t[j] = local_result
            print(f"Completed processing row {j}")

        with Pool(processes=8) as pool:
            for j in range(len(A[:, 0])):
                pool.apply_async(process_row, args=(
                    j, A, Trans, P1, P2, Beliefss, i), callback=callback)
            pool.close()
            pool.join()
            np.save('Data/avg_value_mat_{}'.format(i), avg_t)

    ''' 
    
    '''


    # TODO: line226 may not have the best measure. figure it out!
    # A gives avg tau matrix, for each pair of reward probs and number of distinct policies
    for idx, i in enumerate(T):
        A = np.load(f'Data/avg_value_mat_{i}.npy') / i
        B = np.load(f'Data/distinct_P{i}.npy')
        C = np.load(f'Data/comp_cost_{i}.npy')
        B = np.array(B, dtype=int)

        OV = A[0]
        GV = A[-1]
        CC = np.zeros((len(C), A.shape[1], A.shape[2]))

        dc = C[1] - C[0]
        P1 = np.linspace(0, 1, 50)
        for j in range(len(C)):
            CC[j] = A[B[j]].copy()  # Normalize or scale as needed

        Var_C = np.zeros(A[0].shape)
        print(Var_C.shape)
        for j in range(Var_C.shape[1]):
            for k in range(Var_C.shape[0]):
                Var_C[j, k] = np.sqrt(np.square(np.diff(CC[:, j, k]) / dc).mean())

        fig, ax = plt.subplots()
        cax = ax.imshow(Var_C, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
        ax.set_xlabel('$p_1$')
        ax.set_ylabel(r'$p_2$')

        # Format axes with scientific notation
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_label(r'$\chi_V$')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        plt.tight_layout()
        plt.savefig(f'plots/V_susc_{i}.png', dpi=300)  # Save with high dpi for clarity
        plt.show()
