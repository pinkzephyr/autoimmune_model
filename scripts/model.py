import numpy as np
import parameters as pm
import funcs
from scipy.sparse import lil_matrix
from scipy.sparse import block_diag
import argparse


parser = argparse.ArgumentParser(description='Model probability of autoimmunity given persistent infection.')
parser.add_argument('--t-num', type=int,default=1000000, dest='immature_T_num',
                    help='Number of immature thymocytes, default 1000000')
parser.add_argument('--pep_len', type=int,default=9, dest='pep_len',
                    help='Length of peptides + TCRs, default 9')
parser.add_argument('--N', type=int,default=10000, dest='N',
                    help='Number of self peptides, default 10000')
parser.add_argument('--M-f', type=float,default=0.8, dest='M_f',
                    help='Fraction self peptides presented during selection, default 0.8')
parser.add_argument('--N-T', type=int,default=10, dest='N_T',
                    help='Number of self peptides present in tissue at any given time, default 10')
parser.add_argument('--E-n', type=int,default=-38, dest='E_n',
                    help='Negative energy threshold, default -38 kbt')
parser.add_argument('--N-f', type=int,default=40000, dest='N_f',
                    help='Number of foreign peptides, default 70000')
parser.add_argument('--trials', type=int,default=10000, dest='trials',
                    help='Number of trials conducted to produce probability of autoimmunity, default 10000')
parser.add_argument('--gap', type=float,default=3, dest='gap',
                    help='Gap between E_n and E_self, the lowered energy treshold for foreign-activated T cells, default 3kbt')
parser.add_argument('--quorum', type=int,default=10, dest='quorum',
                    help='Quorum threshold of activation, default 10 activated T cells')
parser.add_argument('--alpha', type=int,default=1, dest='alpha',
                    help='Slope of activation curve, default 1')
parser.add_argument('--thres', type=float,default=0.5, dest='thres',
                    help='Threshold of activation curve, default 0.5')                    
parser.add_argument('--frac', type=float,default=0.04, dest='frac',
                    help='Fraction of thymocytes that survive selection, default 0.04')      
parser.add_argument('--num-set', type=int,default=1, dest='num_set',
                    help='default 1')              
parser.add_argument('--self-pep', type=str, default='self_peptides.npy',
                    help='Self peptide set')
parser.add_argument('--for-pep', type=str, default='foreign_peptides.npy',
                    help='Foreign peptide set')

args = parser.parse_args()
immature_T_num=args.immature_T_num
pep_len=args.pep_len
N=args.N

if args.M_f>1:
    M=int(N*0.1*args.M_f)
else:
    M=int(N*args.M_f)

N_T=args.N_T
E_n=args.E_n
N_f=args.N_f
trials=args.trials
E_gap=args.gap
quorum=args.quorum
alpha=args.alpha
frac=args.frac
thres=args.thres
num_set=args.num_set

self_pep_name=args.self_pep
for_pep_name=args.for_pep

E_new=E_n
E_self=E_new+E_gap
results={}

#generating thymocytes
MJ_20 = block_diag(list(pm.J_mat for i in range(pep_len)))
naive_thymo = funcs.genRepSet(immature_T_num, pm.hum_freq_list, pep_len)
A = lil_matrix((naive_thymo.shape[0], 20*pep_len))
for i in range(naive_thymo.shape[0]):
    for j in range(pep_len):
        A[i,20*j+int(naive_thymo[i,j])] = 1
A = A.tocsr()

#getting self and foreign peptides
self_rep_tot=np.load(self_pep_name, allow_pickle=True)
rand_self_idx=np.random.choice(np.shape(self_rep_tot)[0], N,replace=False)
self_rep=funcs.conv_rep(self_rep_tot[rand_self_idx,:])

for_rep_tot=np.load(for_pep_name, allow_pickle=True)
rand_for_idx=np.random.choice(np.shape(for_rep_tot)[0], N_f,replace=False)
for_rep=funcs.conv_rep(for_rep_tot[rand_for_idx,:])
print('got for and self')

#thymic selection
thymo_min_energy = funcs.selection(A, MJ_20, self_rep, M)

gap_range = np.arange(0.01, 10, 0.001)  # range of gaps to try out
p_sel_out = np.array(
        [np.logical_and(thymo_min_energy < E_n + gap, thymo_min_energy > E_n).mean() for gap in gap_range])
gap_idx = np.argmin(np.abs(p_sel_out - frac))
gap = gap_range[gap_idx] 
        
effector_T_idx = np.logical_and(thymo_min_energy < E_n + gap, thymo_min_energy > E_n)
effector_T = A[effector_T_idx, :]

act_self_eff, act_energies=funcs.pep_exposure(np.shape(self_rep)[0],self_rep, effector_T, alpha,  np.shape(self_rep)[0], E_new, MJ_20, thres)
act_for_eff, for_energies=funcs.pep_exposure(np.shape(for_rep)[0],for_rep, effector_T, alpha, np.shape(for_rep)[0], E_new, MJ_20, thres)

act_self_weak, act_weak_energies=funcs.pep_exposure(np.shape(self_rep)[0],self_rep, effector_T, alpha,  np.shape(self_rep)[0], E_self, MJ_20, thres)
act_for_weak, act_weak_energies=funcs.pep_exposure(np.shape(for_rep)[0],for_rep, effector_T, alpha,  np.shape(for_rep)[0], E_self, MJ_20, thres)

foreign_range=range(1,11)
act_self_mat = np.zeros((len(foreign_range), trials))
act_for_mat = np.zeros((len(foreign_range), trials))
act_new_mat = np.zeros((len(foreign_range), trials))
for_ind=0
for for_num in foreign_range:
    for trial_num in np.arange(0,trials):
        random_idx_forT = np.random.choice(np.shape(for_rep)[0], for_num,replace=False)
        act_ind_efor =np.asarray(np.nonzero(act_for_eff[:, random_idx_forT].sum(1))).flatten()

        random_idx_NT = np.random.choice(np.shape(self_rep)[0], N_T,replace=False)
        act_ind_ebefore =np.asarray(np.nonzero(act_self_eff[:,random_idx_NT].sum(1))).flatten()

        if act_ind_efor.size>quorum:
            act_eunique=list(set(act_ind_efor) - set(act_ind_ebefore)) #get indices of tcrs that were activated by foreign but not self
            act_self_small=act_self_weak[act_eunique,:]
            act_ind_eself=np.asarray(np.nonzero(act_self_small[:,random_idx_NT].sum(1))).flatten()
        else:
            act_ind_eself=np.array([])
        act_self_mat[for_ind, trial_num]=act_ind_ebefore.size+act_ind_eself.size
        act_for_mat[for_ind, trial_num]=act_ind_efor.size
        act_new_mat[for_ind, trial_num]=act_ind_eself.size
    for_ind+=1

prob_act_for=np.mean(act_for_mat>quorum, axis=1)
prob_act_self=np.mean(act_self_mat>quorum, axis=1)  
prob_act_new=np.mean(act_new_mat>quorum, axis=1)

results['prob_act_for']=prob_act_for #probability of activation by foreign peptides alone
results['prob_act_self']=prob_act_self #probability of activation by self peptides
results['prob_act_new']=prob_act_new #probability of activation by self peptides after infection
results['foreign_range']=foreign_range
np.save('results'+str(N_f)+'Nf_'+str(M)+'M_'+str(N_T)+'Nt_'+str(trials)+'tri_'+str(float(E_gap))+'gap_'+str(quorum)+'quorum_'+str(alpha)+'alpha_'+str(thres)+'thres'+str(num_set)+'.npy', results)
