'''
In this code we find the optimal meta-BAMDP soln, by assuming that  we only need one computation to change the tides (we obviously check if this is the case).
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import quad
import itertools
from itertools import product, permutations, combinations
from numpy.linalg import matrix_power


'''Helper functions'''

def get_tuple(length, total):
	return list(filter(lambda x:sum(x)==total,product(range(total+1),repeat=length)))

def flatten(xss):
    return [x for xs in xss for x in xs]

def find_indices(arr, values):
    
    conditions = np.isclose(arr, values[0], atol=1e-8)
    for value in values[1:]:
        conditions |= np.isclose(arr, value, atol=1e-8) 
    indices = np.where(conditions)[0]
    return indices



#Finds the optimal base action to be taken in a particular state. 
def policy_opti(state,Beliefs,T):
	##j = time step and state is the state at that time step
	j = int(state.sum())
	Actions = np.ones ((T,T,T,T))
	V = np.zeros ((T+1,T+1,T+1,T+1))
    
	#Bayesian inference rule
	bi = np.eye(4)
	P1 = bi[1]
	P2 = bi[0]
	P3 = bi[3]
	P4 = bi[2]

	for t in  (np.flip(np.arange(j,T))):
		b = Beliefs[t]
		num = len(b[:,0])
          

               
		a1, b1, a2, b2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
		
		p1 = (a1+1.)/(a1+b1+2.)  
		p2 = (a2+1.)/(a2+b2+2.)
		
		
		M_reshaped = b[:, np.newaxis, :]
		K = np.tile(M_reshaped, (1, 4, 1))
		identity_matrix = np.eye(4) 
		K = K+ identity_matrix[np.newaxis, :, :]
		K = K.astype(int)



		K_reshaped = K.reshape(num * 4, 4)
		V_indices = np.ravel_multi_index((K_reshaped[:, 0], K_reshaped[:, 1], K_reshaped[:, 2], K_reshaped[:, 3]), V.shape)
		V_next = V.flat[V_indices].reshape(num, 4)	
                
                
		Va1 = p1[:, np.newaxis]+(1. - p1[:, np.newaxis]) * P1 * V_next + p1[:, np.newaxis] * P2 * V_next
		Va2 = p2[:, np.newaxis]+(1. - p2[:, np.newaxis]) * P3 * V_next + p2[:, np.newaxis] * P4 * V_next
                
                
		val1 = Va1.sum(axis=1)
		val2 = Va2.sum(axis=1)

		thresh = 10**-8                

		indices_val1_gt_val2 = val1 - val2 > thresh
		indices_val2_gt_val1 = val2 - val1 > thresh
		indices_val1_eq_val2 = np.absolute(val1-val2) <= thresh
		
		#print (np.sum(indices_val1_gt_val2),np.sum(indices_val2_gt_val1),np.sum(indices_val1_eq_val2))
			
		V[tuple(b[indices_val1_gt_val2].T)] = val1[indices_val1_gt_val2]
		Actions[tuple(b[indices_val1_gt_val2].T)] = np.zeros (np.sum(indices_val1_gt_val2))

		V[tuple(b[indices_val2_gt_val1].T)] = val2[indices_val2_gt_val1]
		Actions[tuple(b[indices_val2_gt_val1].T)] = np.ones (np.sum(indices_val2_gt_val1))
		
		V[tuple(b[indices_val1_eq_val2].T)] = val2[indices_val1_eq_val2]
		Actions[tuple(b[indices_val1_eq_val2].T)] = 0.5*np.ones (np.sum(indices_val1_eq_val2))

	return Actions[tuple(state.flatten())]


#This function generates the entire policy vector.
def pol(f,Beliefs,T):
    
        
    A_pol = np.zeros ((T,T,T,T))
    for i in range (T):
        for j in range (len(Beliefs[i][:,0])):
            state = Beliefs[i][j]
            a1,b1,a2,b2 = tuple(state)
            p1 = (a1+1.)/(a1+b1+2.)  
            p2 = (a2+1.)/(a2+b2+2.)
            if (f=='greedy'):
                A_pol[tuple(state)] = np.heaviside(p2-p1,0.5)
            else:
                A_pol[tuple(state)] = f(state,Beliefs,T)
           
    return A_pol
    




#Greedy action values (subjective)
def greedy_q(b):

	Va1 = (b[0]+1.)/(b[0]+b[1]+2.)
	Va2 = (b[2]+1.)/(b[2]+b[3]+2.)
	return Va1,Va2

#Given a tree, it finds the subjective value and the action the agent is going to take if terminated thinking.
def subjective_value(s,b_tilde,T, Beliefs):
	t = Beliefs[s].sum()
	#daughters of act 1
	d1s = find_indices(b_tilde[s],np.array([1.1,1.11]))
	
	#daughters of act 2
	d2s = find_indices(b_tilde[s],np.array([1.2,1.21]))
	
	
	va1,va2 = greedy_q(Beliefs[s])
	if (len(d1s) == 0): 
		A = va1*(T-t)

	elif (len(d1s) == 2):
		
		diffy =  Beliefs[d1s[0]]-Beliefs[s]
		if (diffy[0]==0):
			A = va1 * subjective_value(d1s[1],b_tilde,T, Beliefs)[0] + (1.-va1)* subjective_value(d1s[0],b_tilde,T, Beliefs)[0] + va1
		elif (diffy[0]==1):
			A = va1 * subjective_value(d1s[0],b_tilde,T, Beliefs)[0] + (1.-va1)* subjective_value(d1s[1],b_tilde,T, Beliefs)[0] + va1
		
	if (len(d2s) == 0):
		B = va2*(T-t)
		
	elif (len(d2s) == 2):
	
		
		diffy =  Beliefs[d2s[0]]-Beliefs[s]
		if (diffy[2]==0):
			B = va2 * subjective_value(d2s[1],b_tilde,T, Beliefs)[0] + (1.-va2)* subjective_value(d2s[0],b_tilde,T, Beliefs)[0] + va2
		elif (diffy[2]==1):
			B = va2 * subjective_value(d2s[0],b_tilde,T, Beliefs)[0] + (1.-va2)* subjective_value(d2s[1],b_tilde,T, Beliefs)[0] + va2
		
		
	return max(A,B) , (A,B) , np.heaviside (B-A,0.5)
    



# Finds all possible trees, if only 1 computation is allowed (commented part has 2 comps). More generally need to use subg_enumeration file...
def find_2steps(s,trans):
    As = np.array ([1.,2.])
    BB = []
    
    for first_action in As:
        actty = np.array([1+first_action*0.1,1+first_action*0.1+0.01])
        indies = find_indices(trans[s],actty)
        '''
        for i in indies:
            for act in As:
                bb = np.zeros(trans.shape)
                bb[s][indies] = trans[s][indies].copy()
                actty2 = np.array([1+act*0.1,1+act*0.1+0.01])
                indy2 =  find_indices(trans[i],actty2)
                bb[i][indy2] = trans[i][indy2].copy()
                BB.append(bb)
        '''
        ##Now also add 1 step bbs
        bb = np.zeros(trans.shape)
        bb[s][indies] = trans[s][indies].copy()
        BB.append(bb)
    return BB





#This finds the bounded optimal base policy for a computational cost c. The mapping to meta-policy is straightforward as only one computation is needed.
def bounded_val_pol (T,trans,c,Beliefs,wc,dw, pol_reshape, pol_opti):
    #trans the set of all allowed transitions and c cost.
    #T is the final time
    for i in range (len(Beliefs)):
        #print (i)
        if (Beliefs[i].sum() == T):
            #print (i)
            break
    t=i
    polly = np.zeros (t)
    for i in range (len(Beliefs)):
        #print (i)
        if (Beliefs[i].sum() == T+1):
            #print (i)
            break
    t=i
    V = np.zeros (t)
    V_b = np.zeros (t)
    timmy = np.arange(t)
    
    for i in  np.flip(timmy):
        #print (i)
        #Terminal condition
        #print (Beliefs[i].sum(),T)

        if (Beliefs[i].sum() == T):
            V[i] = Beliefs[i][0] + Beliefs[i][2]
            V_b[i] = Beliefs[i][0] + Beliefs[i][2]
        elif (Beliefs[i].sum()<T):
            #print (i,'hoho')
            pw1 , pw2 = greedy_q(Beliefs[i]) 
            pl1 = 1.-pw1
            pl2 = 1.-pw2
            
            greed_act = pol_reshape[i] ##1 if act = 2 
            opt_act = pol_opti[i]
            next_win2 = np.where (trans[i] == 1.21)[0][0]
            next_los2 = np.where (trans[i] == 1.20)[0][0]
            next_win1 = np.where (trans[i] == 1.11)[0][0]
            next_los1 = np.where (trans[i] == 1.10)[0][0]
            
            qa1 = pw1* V[next_win1] + pl1 *V[next_los1]
            qa2 = pw2* V[next_win2] + pl2 *V[next_los2]
            
            VVs = np.array([qa1,0.5*qa1+0.5*qa2,qa2])
            
            
            base_qa1 = pw1* V_b[next_win1] + pl1 *V_b[next_los1]
            base_qa2 = pw2* V_b[next_win2] + pl2 *V_b[next_los2]
            
            base_VVs = np.array([base_qa1,0.5*base_qa1+0.5*base_qa2,base_qa2])
            
            if (i in dw):
                ind_in_wc = np.where (dw==i)[0][0]
                acts = np.array ([greed_act,wc[ind_in_wc][0],wc[ind_in_wc][1]])
                costs = 10**-2*c*np.ones (3)
                costs[0] = 0
                
                vallies = VVs[(2.*acts).astype(int)] - costs
                vallies_b = base_VVs[(2.*acts).astype(int)] 
                
                polly[i] = acts[np.argmax(vallies)]
                V[i] = vallies.max()
                V_b[i] = vallies_b[np.argmax(vallies)]
                if (len(np.where (vallies == vallies.max())[0])>1 ):
                    polly[i] = opt_act
                
            else:
                polly[i] = greed_act
                V[i] = VVs[int(2.*greed_act)]
                V_b[i] = base_VVs[int(2.*greed_act)]
                
    return V, polly,V_b




#Given a base-policy and a game. Find the value of a state i under that policy.
def val_pol (pol,i,T,trans, Beliefs):
    #print (i,T,i+T)
    state = Beliefs[i]
    t = state.sum()

    if (t==T):
        return 0
    
    else:
        act = pol[i]
        if (act == 1):
            next_win = np.where (trans[i] == 1.21)[0][0]
            next_los = np.where (trans[i] == 1.20)[0][0]
            a2 = state[2]
            b2 = state[3]
            p_win = (a2+1.)/(a2+b2+2.)
            return p_win * (1. + val_pol(pol,next_win,T,trans)) + (1.-p_win)*val_pol(pol,next_los,T,trans)
        elif (act==0):
            next_win = np.where (trans[i] == 1.11)[0][0]
            next_los = np.where (trans[i] == 1.10)[0][0]
            a1 = state[0]
            b1 = state[1]
            p_win = (a1+1.)/(a1+b1+2.)
            return p_win * (1. + val_pol(pol,next_win,T,trans)) + (1.-p_win)*val_pol(pol,next_los,T,trans)
        elif (act == 0.5):
            next_win2 = np.where (trans[i] == 1.21)[0][0]
            next_los2 = np.where (trans[i] == 1.20)[0][0]
            next_win1 = np.where (trans[i] == 1.11)[0][0]
            next_los1 = np.where (trans[i] == 1.10)[0][0]
            a1 = state[0]
            b1 = state[1]
            a2 = state[2]
            b2 = state[3]
            p_win2 = (a2+1.)/(a2+b2+2.)
            p_win1 = (a1+1.)/(a1+b1+2.)
            val1 = p_win1 * (1. + val_pol(pol,next_win1,T,trans)) + (1.-p_win1)*val_pol(pol,next_los1,T,trans)
            val2 = p_win2 * (1. + val_pol(pol,next_win2,T,trans)) + (1.-p_win2)*val_pol(pol,next_los2,T,trans)
            return 0.5*val1+0.5*val2
        
        else:
            return 'BC'




def zeroth_rows_of_powers(A, N):
    current_row = A[0, :]  # Start with the 0th row of A
    result = [current_row]  # Store the initial row (A^1)

    for _ in range(1, N):
        current_row = np.dot(current_row, A)
        result.append(current_row)  # Append the result after each multiplication

    return np.array(result)

#Takes the complete transition matrix and a policy and returns the trans_mat under that policy.        
def pol_trans (pol,trans, Beliefs, time):
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
            TT[i][next_win1] = greedy_q(Beliefs[i])[0]
            TT[i][next_los1] = 1. - greedy_q(Beliefs[i])[0]
        elif (act == 1):
            TT[i][next_win1] = 0
            TT[i][next_los1] = 0
            TT[i][next_win2] = greedy_q(Beliefs[i])[1]
            TT[i][next_los2] = 1. - greedy_q(Beliefs[i])[1]
        elif (act == 0.5):
            TT[i][next_win2] = 0.5*greedy_q(Beliefs[i])[1]
            TT[i][next_los2] = 0.5*(1. - greedy_q(Beliefs[i])[1])
            TT[i][next_win1] = 0.5*greedy_q(Beliefs[i])[0]
            TT[i][next_los1] = 0.5*(1. - greedy_q(Beliefs[i])[0])
    T_pow  =    zeroth_rows_of_powers(TT, time-1)         
    return TT,T_pow
    
#Takes the policy trans_mat and finds the probability of transitioning from i to j
def trans_prob(TT,i,j):
    ti = Beliefs[i].sum()
    tj = Beliefs[j].sum()    
    return matrix_power (TT,tj-ti)[i,j]


def generate_data(time):
    #For some upper limit we generate all possible beliefs
    T = 20
    Beliefs = []
    for i in range (T+1):
        Beliefs.append(np.asarray(get_tuple(4,i)))
        


    #Actual time of the game considered    
    # time = int(input('T='))
    print ('Length of task:',time)
    Pol_greed = pol('greedy',Beliefs,time)
    Pol_opti = pol(policy_opti,Beliefs,time)

    #Now convert the policy array into a linear vector for the natural state ordering
    N = len(flatten(Beliefs[:time]))
    pol_reshape = np.zeros (N)
    pol_opti = np.zeros (N)

    for i in range (N):
        state = flatten(Beliefs[:time])[i]
        pol_reshape[i] = Pol_greed [tuple(state)]
        pol_opti[i] = Pol_opti[tuple(state)]

    #Renaming for arbitary reason
    A =pol_reshape


    #Now we make the complete transition matrix of all allowed transitions. Not just the reachable ones under a policy. This just has the structure of the task.
    Trans = np.zeros ((len(flatten(Beliefs[:time+1])),len(flatten(Beliefs[:time+1]))))
    for i in range (len(A)):
        bi = flatten(Beliefs)[i]
        K = np.tile(bi, (1, 4,1))
        identity_matrix = np.eye(4) 
        K = K+ identity_matrix

        for j in range (4):
            indy = np.where((flatten(Beliefs) == K[0][j]).all(axis=1))[0]
            veccy = np.array([1.11, 1.1, 1.21, 1.2])
            #if (indy>len(Trans)):
                #break
            #else:
            Trans [i,indy] = veccy[j]

    #Change of Beliefs shape for an arbitrary reason..
    Beliefs = np.vstack(Beliefs)

    #Find all the states where 1 comp can change it all
    doesnt_work = []
    with_comps = []
    for s in range (len(A)):

        b_tilde = np.zeros (Trans.shape)
        #This is what the agent would do without any comp
        ans = subjective_value(s,b_tilde,time, Beliefs)[2]
        
        #Find possible beliefs after comps
        compys = find_2steps(s,Trans)
        ANSC = []
        for i in range(len(compys)):
            b_try = compys[i]
            #Answer after comp
            anst = subjective_value(s,b_try,time, Beliefs)[2]
            ANSC.append(anst)
        
        #Do the answers before and after comp match? If not, then 1 comp can flip it.
        if not(np.all(np.array(ANSC)==ans)):
            doesnt_work .append(s)
            with_comps .append(ANSC)
            
    #print (doesnt_work,np.where(pol_opti != A)[0])
    #print (np.intersect1d((np.intersect1d(np.where(pol_opti != A)[0],doesnt_work)),(np.where(pol_opti != A)[0])))
    #print (A[np.intersect1d(np.where(pol_opti != A)[0],doesnt_work)])


    ##This prints true if the convex hull can be flipped in 1 comp, else false. If false, try smaller time steps.
    print ('Are the states in the convex hull flippable in 1 step?',np.isin(np.where(pol_opti != A)[0], doesnt_work).all())




    #Greedy and optimal policies renamed (once again!)
    GP = pol_reshape
    OP = pol_opti

    #Prints the size of the convex hull
    print ('Size of the convex hull, ignoring reachability:',len(np.where (GP!=OP)[0]))


    #Consider a range of computational costs
    costies = np.linspace (0,16,2000)
    vallies = costies.copy()
    comp_nums = costies.copy()
    avg_t = costies.copy()
    all_polls = []
    distinct_pols = 0
    distinct_P = np.zeros (len(costies))
    for i in range (len(costies)):


        sollito = bounded_val_pol (time,Trans,costies[i],Beliefs,with_comps,doesnt_work, pol_reshape, pol_opti)
        BP = sollito[1]
        base_vally = sollito[2]
        #print (base_vally[0])
        

        #plt.plot (costies[i],len((np.where (BP!=OP)[0])),'ko')
        
        
        
        ##Some more effort for finding average time
        avt = 0
        if (i>0) and (np.all(BP==bpl)):
            #print (i)
            avg_t[i] = avg_t[i-1]
            vallies[i] = vallies[i-1]
            comp_nums[i] = comp_nums[i-1]
            distinct_P[i] = distinct_P[i-1]
            continue
        print (i)
        distinct_pols+=1
        distinct_P[i] = distinct_pols-1
        all_polls.append(BP)
        inddy = np.where (BP!=GP)[0]
        TT,T_pow = pol_trans (BP,Trans, Beliefs, time)
        bpl = BP.copy()
        for j in inddy:
            #print (j)
            avt+=Beliefs[j].sum()*T_pow[Beliefs[j].sum()-1][j] 
        
        avg_t[i] = avt
        vallies[i] = base_vally[0]
        comp_nums[i] = (vallies[i]-sollito[0][0])/costies[i]*10**2	##Scaling bndd_vl_pl
        
    all_polls = np.array(all_polls)    
    print (len(all_polls[:,0]),np.unique (distinct_P))
    np.save('Data/all_polls_{}'.format(time),all_polls)     
    np.save('Data/avg_comp_time_{}'.format(time),avg_t) 
    np.save('Data/total_exp_rew_{}'.format(time),vallies)
    np.save('Data/avg_comp_nums_{}'.format(time),comp_nums)
    np.save('Data/comp_cost_{}'.format(time),costies)
    np.save('Data/distinct_P{}'.format(time),distinct_P)
    
        
    # plt.plot (costies,avg_t,'ko')
    # plt.plot (costies,distinct_P)
    #plt.plot (costies,val_pol (GP,0,time,Trans)*np.ones(len(costies)))
    #plt.plot (costies,val_pol (OP,0,time,Trans)*np.ones(len(costies)))
    # plt.show()

    #print (BP[np.where (OP!=BP)[0]])
    #print (GP[np.where (OP!=BP)[0]])
    #print (OP[np.where (OP!=BP)[0]])