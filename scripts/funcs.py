import numpy as np
from scipy.sparse import lil_matrix
import multiprocessing
from functools import partial

#Functions for creating immature thymocyte sets
def convPeptide(peptide, conversion, aaTonum=True):
    # converts peptide into indices for matrix
    # input: peptide: peptide in question
    #        conversion: conversion from alpha->num
    #        aaTonum: whether it is from alphabet to numerical or other way around (aaTonum=True means from alph->num)
    indice_list = []
    for i in peptide:
        if aaTonum == True:
            if i in conversion:
                indice = conversion.index(i)
                indice_list.append(indice)
        else:
            indice = conversion[int(i)]
            indice_list.append(indice)
    if aaTonum==False:
        indice_list=''.join(indice_list)
    return indice_list

def convPeptideList(txtfile, outfile, conversion, aaTonum=True):
    #converts list of peptides either to numerical or alphabetical
    #input: txtfile: file that will be converted
    #       outfile: name of output file
    #       conversion: conversion from numberical to alphabetical
    #       aaTonum: whether it is from alphabet to numerical or other way around (aaTonum=True means from alph->num)
    #output:conv_list: list of converted peptides
    #       also autosaves new list
    pep_list = np.loadtxt(txtfile, delimiter=' ', dtype='str')
    conv_list = []
    for i in pep_list:
        i=str(i)
        conv_i = convPeptide(i, conversion, aaTonum)
        conv_list.append(conv_i)
    return conv_list


def conv_rep(load_rep, pep_len):
    # converts loaded txt file of peptides into sparse matrix
    # inputs: load_rep:  loaded rep directly from txt
    #         filename: name of file
    #         pep_len: length of peptides in question
    # output: A: converted sparse rep
    A = lil_matrix((load_rep.shape[0], 20 * load_rep.shape[1]))
    for i in range(A.shape[0]):
        for j in range(np.shape(load_rep)[1]):
            try:
                A[i, 20 * j + load_rep[i, j]] = 1
            except IndexError as e:
                print(e,20 * j + load_rep[i, j])
                exit()
            A[i, 20 * j + load_rep[i, j]] = 1  # row is the peptide, column is the peptide in specific spot
    return A

def A_to_list(A, pep_len):
# converts sparse matrix of peptides into numpy array
# inputs: A:  matrix of peptides
#         pep_len: length of peptides in question
# output: pep_list: numpy array of peptides
    # A = lil_matrix((load_rep.shape[0], 20 * pep_len))
    pep_list=np.zeros((A.shape[0],int(A.shape[1]/20.0)))
    pep_len=int(A.shape[1]/20.0)
    tot_list = A.nonzero()[1]
    for i in range(A.shape[0]):
        pep=np.zeros(pep_len)
        raw_pep=tot_list[pep_len*i:pep_len*i+pep_len]
        for j in range(pep_len):
            aa_idx=raw_pep[j]-j*20
            pep[j]=aa_idx
        pep_list[i, :] = pep
    return pep_list

def genRepSet(numPep_val, freq_list, N_val):
# Generates a repertoire set randomly given frequency weights
# inputs: numPep_val = number of peptides that are to be generated in the collection
#         freq_list = frequency of peptides
#         N_val = length of peptide
# outputs: randPeptide_array: array of randomly generated peptides
    randPeptide_array = np.zeros([numPep_val, N_val])
    for i in np.arange(numPep_val):
        randPeptide = np.random.choice(int(len(freq_list)), N_val, p=freq_list)
        randPeptide_array[i, :] = randPeptide
    return randPeptide_array

#Functions for selection

def min_calc(N, M, pep_set, MJ_20, A, i):
    #finds strongest energy interaction between one thymocyte and group of peptides
    #inputs: N = number of peptides to choose from 
    #        M = number of self peptides used during selection
    #        pep_set = immature thymocyte repertoire
    #        MJ_20 = MJ_20 = Miyazawa Jerningan matrix for interaction energies between peptides
    #        A = immature thymocyte repertoire
    #        i = immature thymocyte exposed to self peptides
    # output: strongest binding energy between thymocyte and peptides

    self_idx = np.sort(np.random.choice(np.arange(N), M, replace=False))  # selection number
    energies = A[i, :].dot(MJ_20.dot(pep_set[self_idx, :].T))  # vector of energies for ith tcr
    return energies.min()



def selection(A, MJ_20, pep_set, N, M):
    # perform selection on thymocyte group
    # input: HLA rep matrix, self pep matrix, MJ matrix, HLA rep matrix, filename of HLA matrix
    # output: thymo_min_energy = list of minimum energies

    a_pool = multiprocessing.Pool()
    func = partial(min_calc, N, M, pep_set, MJ_20, A)
    temp_ = a_pool.map(func, range(A.shape[0]))
    a_pool.close()
    a_pool.join()

    thymo_min_energy = np.array(temp_)
    return thymo_min_energy

#Functions to determine activated thymocytes

def activation_func(alpha, E_n, E_i):
    #vectorized function that takes in E_i between peptide and tcr, outputs probability of activation
    #inputs: alpha = slope of activation curve
    #        E_n = negative selection threshold (used to set activation curve)
    #        E_i = interaction energy
    #output: scored interaction energy
    
    return 1/(1 + np.exp(-(E_n-E_i)/alpha))

def pep_exposure(pep_num,pep_set, thymo_rep, alpha, N, E_n, MJ_20, thres):
    #returns a matrix of activated t cells given a set of self peptides
    #inputs: pep_num = number of peptides to expose repertoire
    #        pep_set = set of peptides to choose from
    #        thymo_rep = set of thymocytes
    #        alpha = slope of activation curve
    #        N = number of peptides to choose from
    #        E_n = negative selection threshold (used to set activation curve)
    #        MJ_20 = Miyazawa Jerningan matrix for interaction energies between peptides
    #        thres = threshold to determine which thymocytes are activated
    #output: act_mat: matrix with 0 for peptide not activating thymocyte, 1 for peptide activating thymocyte 
    random_idx = np.sort(np.random.choice(np.arange(N), pep_num, replace=False))
    pep_set = pep_set[random_idx, :]
    energies = (thymo_rep.dot(MJ_20.dot(pep_set.T))).todense()
    prob_mat = activation_func(alpha, E_n, energies)
    act_mat = np.asarray(np.greater(prob_mat, thres))
    return act_mat, energies


def pep_exposure_given(pep_set, thymo_rep, alpha, E_n, MJ_20, thres):
    #returns a matrix of activated t cells given a set of self peptides
    energies = (thymo_rep.dot(MJ_20.dot(pep_set.T))).todense()
    prob_mat = activation_func(alpha, E_n, energies)
    act_mat = np.asarray(np.greater(prob_mat, thres))
    return act_mat, energies

    



