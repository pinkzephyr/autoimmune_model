import numpy as np
import parameters as pm
import utils as funcs
from scipy.sparse import lil_matrix
from scipy.sparse import block_diag
import argparse
import os

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
parser.add_argument('--frac', type=float,default=0.04, dest='frac',
                    help='Fraction of thymocytes that survive selection, default 0.04')      
parser.add_argument('--num-set', type=int,default=1, dest='num_set',
                    help='default 1')              
parser.add_argument('--self-pep', type=str, default='self_peptides.npy',
                    help='Self peptide set')
parser.add_argument('--for-pep', type=str, default='foreign_peptides.npy',
                    help='Foreign peptide set')
parser.add_argument('--output', type=str, default='',
                    help='directory for output')


args = parser.parse_args()
immature_T_num=args.immature_T_num
pep_len=args.pep_len
N=args.N

if args.M_f>1:
    M=int(N*0.1*args.M_f)
else:
    M=int(N*args.M_f)

N_T=args.N_T

if args.E_n>0:
    E_n=-args.E_n
else:
    E_n=args.E_n
N_f=args.N_f
trials=args.trials
E_gap=args.gap
quorum=args.quorum
frac=args.frac
num_set=args.num_set

self_pep_name=args.self_pep
for_pep_name=args.for_pep

out_dir=args.output

E_self=E_n+E_gap
results={}

#getting self and foreign peptides
self_pep_name=os.path.join(os.path.dirname(__file__), '..', 'data', self_pep_name)
self_rep_tot=np.load(self_pep_name, allow_pickle=True)
self_rep=funcs.conv_rep(self_rep_tot[:N,:], pep_len)

for_pep_name=os.path.join(os.path.dirname(__file__), '..', 'data', for_pep_name)
for_rep_tot=np.load(for_pep_name, allow_pickle=True)
for_rep=funcs.conv_rep(for_rep_tot[:N_f,:], pep_len)

#A=A.astype('float32')
self_rep=self_rep.astype('float32')
for_rep=for_rep.astype('float32')

#generating thymocytes
MJ_20 = block_diag(list(pm.J_mat for i in range(pep_len)))


if M==8000:
    
    naive_thymo = funcs.genRepSet(immature_T_num, pm.hum_freq_list, pep_len)
    A = lil_matrix((naive_thymo.shape[0], 20*pep_len))
    for i in range(naive_thymo.shape[0]):
        for j in range(pep_len):
            A[i,20*j+int(naive_thymo[i,j])] = 1
    A = A.tocsr()

    thymo_min_energy = funcs.selection(A, MJ_20, self_rep, N, M)

    gap_range = np.arange(0.01, 10, 0.001)  # range of gaps to try out
    p_sel_out = np.array(
            [np.logical_and(thymo_min_energy < E_n + gap, thymo_min_energy > E_n).mean() for gap in gap_range])
    gap_idx = np.argmin(np.abs(p_sel_out - frac))
    gap = gap_range[gap_idx] 
else: #if not the base case, then need to find the number of immature thymocytes to keep the gap the same size and 4% of thymocytes
    gap=0.47 #gap between En and Ep for M=8000
    thymo_range = np.arange(500000,1000000,50000)
    p_sel_out=[]
    for immature_T_num in thymo_range:
        naive_thymo = funcs.genRepSet(immature_T_num, pm.hum_freq_list, pep_len)
        A = lil_matrix((naive_thymo.shape[0], 20*pep_len))
        for i in range(naive_thymo.shape[0]):
            for j in range(pep_len):
                A[i,20*j+int(naive_thymo[i,j])] = 1
        A = A.tocsr()
        A=A.astype('float32')
        thymo_min_energy = funcs.selection(A, MJ_20, self_rep, N, M)
        p_sel_out.append((np.logical_and(thymo_min_energy < E_n + gap, thymo_min_energy > E_n)).mean())
    thymo_idx=np.argmin(np.abs(np.array(p_sel_out) - frac))
    immature_T_num=thymo_range[thymo_idx]

    naive_thymo = funcs.genRepSet(immature_T_num, pm.hum_freq_list, pep_len) #get actual repertoire
    A = lil_matrix((naive_thymo.shape[0], 20*pep_len))
    for i in range(naive_thymo.shape[0]):
        for j in range(pep_len):
            A[i,20*j+int(naive_thymo[i,j])] = 1
    A = A.tocsr()
    A=A.astype('float32')
    thymo_min_energy = funcs.selection(A, MJ_20, self_rep, N, M)

effector_T_idx = np.logical_and(thymo_min_energy < E_n + gap, thymo_min_energy > E_n)
effector_T = A[effector_T_idx, :]

#for ease of using self/foreign peptide repertoires:
self_rep_num=np.shape(self_rep)[0]
for_rep_num=np.shape(for_rep)[0]

#get the number of activated T cells from self/foreign peptide repertoires
act_self_eff, act_energies=funcs.pep_exposure(self_rep_num,self_rep, effector_T, self_rep_num, E_n, MJ_20)
act_for_eff, for_energies=funcs.pep_exposure(for_rep_num,for_rep, effector_T, for_rep_num, E_n, MJ_20)

results['act_self']=np.mean(act_self_eff.sum(1)/float(np.shape(act_self_eff)[1]))
results['act_for']=np.mean(act_for_eff.sum(1)/float(np.shape(act_for_eff)[1]))

act_self_weak, act_weak_energies=funcs.pep_exposure(self_rep_num,self_rep, effector_T, self_rep_num, E_self, MJ_20)
act_for_weak, act_weak_energies=funcs.pep_exposure(for_rep_num,for_rep, effector_T, for_rep_num, E_self, MJ_20)

#modelling infection
foreign_range=[0,1,2,3,4,5,6,7,8,9,10]
for_num_delta=[0,1,1,1,1,1,1,1,1,1,1]

act_self_mat = np.zeros((len(foreign_range), trials))
act_for_mat = np.zeros((len(foreign_range), trials))
act_new_mat = np.zeros((len(foreign_range), trials))

self_act_dist_tot=[]
for_act_dist_tot=[]
for trial_num in np.arange(0,trials):
    self_act_dist=[]
    for_act_dist=[]
    random_idx_NT = np.random.choice(self_rep_num, N_T,replace=False)
    act_ind_ebefore =np.asarray(np.nonzero(act_self_eff[:,random_idx_NT].sum(1))).flatten()
    self_act_dist.append(act_ind_ebefore.size)

    for_indices=np.array([], dtype='int64')
    
    for_num=0
    for Nf_num in foreign_range:
        idx=np.random.choice(for_rep_num, for_num_delta[for_num],replace=False)
        for_indices = np.append(idx, for_indices)
        act_ind_efor =np.asarray(np.nonzero(act_for_eff[:, for_indices].sum(1))).flatten()
        for_act_dist.append(act_ind_efor.size)

        if act_ind_efor.size>quorum:
            
            act_eunique=list(set(act_ind_efor) - set(act_ind_ebefore)) #get indices of tcrs that were activated by foreign but not self
            act_self_small=act_self_weak[act_eunique,:]
            act_ind_eself=np.asarray(np.nonzero(act_self_small[:,random_idx_NT].sum(1))).flatten()
            print(Nf_num, act_ind_efor.size, act_ind_eself.size)
        else:
            act_ind_eself=np.array([])
        act_self_mat[for_num, trial_num]=act_ind_ebefore.size+act_ind_eself.size
        act_for_mat[for_num, trial_num]=act_ind_efor.size
        act_new_mat[for_num, trial_num]=act_ind_ebefore.size
        for_num+=1
    
    self_act_dist_tot.append(self_act_dist)
    for_act_dist_tot.append(for_act_dist)

prob_act_for=np.mean(act_for_mat>quorum, axis=1)
prob_act_self=np.mean(act_self_mat>quorum, axis=1)  
prob_act_new=np.mean(act_new_mat>quorum, axis=1)

results['self_eff_avg']=np.mean(act_self_eff.sum(1)/float(np.shape(act_self_eff)[1]))
results['for_eff_avg']=np.mean(act_for_eff.sum(1)/float(np.shape(act_for_eff)[1]))

results['self_eff_weak_avg']=np.mean(act_self_weak.sum(1)/float(np.shape(act_self_weak)[1]))
results['for_eff_weak_avg']=np.mean(act_for_weak.sum(1)/float(np.shape(act_for_weak)[1]))

results['prob_act_for']=prob_act_for #probability of activation by foreign peptides alone
results['prob_act_self']=prob_act_self #probability of activation by self peptides
results['prob_act_new']=prob_act_new #probability of activation by self peptides after infection
results['foreign_range']=foreign_range

np.save(out_dir+'//results'+str(N_f)+'Nf_'+str(M)+'M_'+str(N_T)+'Nt_'+str(trials)+'tri_'+str(float(E_gap))+'gap_'+str(quorum)+'quorum_'+str(num_set)+'.npy', results)
