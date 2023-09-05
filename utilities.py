import numpy as np
import json
import pickle
from collections import OrderedDict


def load_inputData(dataName):
    if dataName=="Davis":
        ligands = json.load(open("./data/Davis_data/davis_ligands.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open("./data/Davis_data/davis_proteins.txt"), object_pairs_hook=OrderedDict)
    elif dataName=="Kiba":
        ligands = json.load(open("./data/Kiba_data/kiba_ligands.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open("./data/Kiba_data/kiba_proteins.txt"), object_pairs_hook=OrderedDict)
    else:
        ligands = json.load(open("./data/BindDB_data/BindDB_ligands.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open("./data/BindDB_data/BindDB_proteins.txt"), object_pairs_hook=OrderedDict)
       

    drugs = []
    for l in ligands.keys():
      drugs.append(ligands[l])

    targets = []
    for p in proteins.keys():
      targets.append(proteins[p])
      
    return drugs, targets  

def load_outputData(dataName): 
    if dataName=="Davis":
        Kd = np.loadtxt("./data/Davis_data/Y_davis.txt")    
        PKd = -(np.log10(Kd/(1e9))) 
    else:
        PKd = pickle.load(open("./data/Kiba_data/Y_kiba","rb"), encoding='latin1') 
        
    return PKd


def smile_2_words(smile, q):
    words_List = []

    ls = len(smile)	
    for index in range(ls -(q-1)):
        word8 = smile[index:index+q]
        words_List.append(word8)
    return words_List	
			
def seq_2_words(seq, q):
    P_words_List = []
    ls = len(seq)
	
    for index in range(ls -(q-1)):
        word3 = seq[index:index+q]
        word3 = word3.upper()              
        P_words_List.append(word3)
    return P_words_List


# binarization (1,0) 
def kiba_Y_binary(Y):
    threshold_val = 12.1
    Y_binary = np.where(Y >= threshold_val, 1., 0.)
    return Y_binary

def davis_Y_binary(Y):
    threshold_val = 7.0
    Y_binary = np.where(Y >= threshold_val, 1., 0.)
    return Y_binary

def bindDB_Y_binary(Y):
    thval = 7.0
    Y_binary =np.where(Y >= thval, 1., 0.)
    return Y_binary


def oneHot_encoding(line, max_sequence_len, comb_wrd_ind):
    X = np.zeros((max_sequence_len, len(comb_wrd_ind)))
    for i, ch in enumerate(line[:max_sequence_len]):
        X[i, (comb_wrd_ind[ch])-1] = 1 

    return X 

def label_encoding(line, max_sequence_len, comb_wrd_ind):
	X = np.zeros(max_sequence_len)
  
	for i, ch in enumerate(line[:max_sequence_len]):
		X[i] = comb_wrd_ind[ch]
    
	return X 
   
####   encoded data  (one-hot / label)  

def oneHot_OR_catigorical_data(drugProtein, DT_chars_dict, max_sequence_len): 
    # drugProtein: as it can be either DRUG OR PROTEIN
    X = []
    with_label=True
    
    if with_label:
        for d in range(len(drugProtein)):
            X.append(label_encoding(drugProtein[d], max_sequence_len, DT_chars_dict))
    else:
        for d in range(len(drugProtein)):
            X.append(oneHot_encoding(drugProtein[d], max_sequence_len, DT_chars_dict))
  
    return X


def get_save_paths(hyperparameter_configuration, dataName, v):
    save_path = f"./results/{dataName}_("
    for index, (hyperparameter_name, hyperparameter_value) in enumerate(hyperparameter_configuration.items()):
        save_path += hyperparameter_name
        save_path += "_"
        save_path += str(hyperparameter_value)
        if index != len(hyperparameter_configuration) - 1:
            save_path += ","
    save_path += ")"

    
    model_save_path = save_path + "-v"
    model_save_path += str(v)
    model_save_path += ".h5"

    c_index_plot_save_path = save_path + "_cindex-v"
    c_index_plot_save_path += str(v)
    c_index_plot_save_path += ".png"

    mse_plot_save_path = save_path + "_mse-v"
    mse_plot_save_path += str(v)
    mse_plot_save_path += ".png"

    qq_plot_save_path = save_path + "_QQplot_validationData-v"
    qq_plot_save_path += str(v)
    qq_plot_save_path += ".png"


    return model_save_path, c_index_plot_save_path, mse_plot_save_path, qq_plot_save_path

