from utilities import load_inputData, smile_2_words, seq_2_words, load_outputData, davis_Y_binary, kiba_Y_binary, oneHot_OR_catigorical_data, bindDB_Y_binary

import numpy as np
import pandas as pd
import multiprocessing
import json
from gensim.models import Word2Vec
from collections import OrderedDict


class  Drug_Sequence(object):
    def __init__(self,dirname):
        self.dirname = dirname
        pass
    
    def __iter__(self):
        
        for index, smile in enumerate(self.dirname):
            l = len(smile)
            if l < 1:
                while l < 1:
                    smile = smile + "_"
                    l = len(smile)
                     
            smile_words = smile_2_words(smile, 1)
            yield smile_words

							
class Protein_Sequence(object):
    def __init__(self,dirname):
        self.dirname = dirname
        pass
    
    def __iter__(self):
        
        for index, seq in enumerate(self.dirname):
            l= len(seq)
            if l < 1:
                while l < 1:
                    seq = seq + "_"
                    l = len(seq)               
      
            seq_words = seq_2_words(seq, 1)
            yield seq_words	


def preprocessing_Drugs(drugs_):
    
    ## seq ---> list of chars
    D_Sequence = Drug_Sequence(drugs_)    


    ## Training using Word2Vec
    D_model = Word2Vec(D_Sequence, vector_size=128, window=10, min_count=1, sample=1e-4, negative=5, epochs=20, sg=1, hs=0, workers=multiprocessing.cpu_count())


    
    ### Create Vocabular dictionary
    D_vocab= sorted(list(D_model.wv.key_to_index))
    

    # extract tokens
    D_tokens_ = []
    for d in D_Sequence:
        D_tokens_.append(d)

    # create vocabular dict
    all_chars= sum(D_tokens_, [])
    chars_set = sorted(list(set(all_chars)))
    chars_2_int = dict((c,i+1) for i,c in enumerate(chars_set))

    
    ## Including all 1grams in chars_2_int dict
    count = len(chars_2_int)
    for _, item in enumerate(D_vocab):
        if item not in chars_2_int:
            count = count + 1
            chars_2_int[item] = count
        

    return D_tokens_, chars_2_int


def preprocessing_Proteins(proteins_):
  

    ## seq ---> list of chars
    P_Sequence = Protein_Sequence(proteins_) 


    ## Training using Word2Vec
    P_model = Word2Vec(P_Sequence, vector_size=128, window=10, min_count=1, sample=1e-4, negative=5, epochs=20, sg=1, hs=0, workers=multiprocessing.cpu_count())
 
    ### Create Vocabular dictionary
    P_vocab= sorted(list(P_model.wv.key_to_index))


    # extract tokens
    P_tokens_= []
    for p in P_Sequence:
        P_tokens_.append(p)    
    


    # create Vocabular dictionary
    all_chars= sum(P_tokens_, [])
    chars_set = sorted(list(set(all_chars)))
    chars_2_int = dict((c,i+1) for i,c in enumerate(chars_set))
    
    ## including all 1grams in chars_2_int dict
    count = len(chars_2_int)
    for _, item in enumerate(P_vocab):
        if item not in chars_2_int:
            count = count + 1
            chars_2_int[item] = count

    return P_tokens_, chars_2_int

def CrossValidation(DatasetName):
    
    # load 6 folds (test_set, train_set_5)
    if DatasetName=="Davis":
        test_set = json.load(open("./data/Davis_data/davis_testFolds.txt"))  
        train_set_5 = json.load(open("./data/Davis_data/davis_trainFolds.txt"))
    if DatasetName=="Kiba": 
        test_set = json.load(open("./data/Kiba_data/kiba_testFolds.txt"))  
        train_set_5 = json.load(open("./data/Kiba_data/kiba_trainFolds.txt"))
     

    fold_5 = len(train_set_5)

    

    ## TRAIN/VAL/TEST datasets
    test_sets = []
    val_sets = []
    train_sets = []

    for val_foldind in range(fold_5):

        val_fold = train_set_5[val_foldind]
        val_sets.append(val_fold)  
        
        otherfolds = list(range(fold_5))
        otherfolds.pop(val_foldind)  
        otherfoldsinds = [item for i in otherfolds for item in train_set_5[i]]
         
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set) 
            
    
    # print(" number of folds for training_data: ", len(train_sets))  = 5 
    # print(" number of folds for validation_data: ", len(val_sets))  = 5 
    # print(" number of folds for test_data: ", len(test_sets))       = 5
    return train_sets, val_sets, test_sets 


def prepare_interaction_pairs(DataName, XD, XP, Y, rows, cols):
    Fdrugs = []
    Fproteins = []
    Yaffinity=[] 
    #print("nbre de training inputs:", len(rows)) 
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        Fdrugs.append(drug)

        protein = XP[cols[pair_ind]]
        Fproteins.append(protein)
        
        Yaffinity.append(Y[rows[pair_ind],cols[pair_ind]])
    
    # binaryzation
    if DataName == "Davis":
        Y_binary = davis_Y_binary(np.asarray(Yaffinity))
    elif DataName == "Kiba":
        Y_binary = kiba_Y_binary(np.asarray(Yaffinity))
    
    return Fdrugs, Fproteins, Y_binary, Yaffinity

def final_inputs_output(DataName, Dr, Pr, Affinity, label_row_inds, label_col_inds, train_sets, val_sets, test_sets, v):   
    
        valinds = val_sets[v]
        trainsets_inds = train_sets[v]
        testsets_inds = test_sets[v]

        
        # Training data
        trrows = label_row_inds[trainsets_inds]
        trcols = label_col_inds[trainsets_inds]
        
        train_drugs, train_prots, train_Y, train_Y_aff = prepare_interaction_pairs(DataName, Dr, Pr, Affinity, trrows, trcols)

        
        #  Validation data
        varows = label_row_inds[valinds]
        vacols = label_col_inds[valinds]
        val_drugs, val_prots, val_Y, val_Y_aff = prepare_interaction_pairs(DataName, Dr, Pr, Affinity, varows, vacols)

        #  test data
        terows = label_row_inds[testsets_inds]
        tecols = label_col_inds[testsets_inds]
        test_drugs, test_prots, test_Y, test_Y_aff = prepare_interaction_pairs(DataName, Dr, Pr, Affinity, terows, tecols)
        
        return train_drugs, train_prots, train_Y, train_Y_aff, val_drugs, val_prots, val_Y, val_Y_aff, test_drugs, test_prots, test_Y, test_Y_aff
  

def final_TrainingValidationTest_data(DataName, Dtokens, Ptokens, D_char2int, P_char2int, max_sequence_len):
 
    D = Dtokens # input
    P = Ptokens # input
    Y_Affi = load_outputData(DataName) # output (regression)
 
    D = np.asarray(D)    
    P = np.asarray(P)    
    Y_Affi = np.asarray(Y_Affi)  

    
    # print("Number of drugs:", D.shape[0])  # 68
    # print("Number of proteins:", P.shape[0])  # 442

    
    # finds the indices [i,j] of each affinity  
    label_row_inds, label_col_inds = np.where(np.isnan(Y_Affi)==False)    

    

    # 5-fold Cross-validation  
    train_set, val_set, test_set = CrossValidation(DataName) 
    
    
    
    ########      input / output    data 
    ########    tr_rows = tr_cols = 20036
    ########    va_rows = va_cols =  5010
    

    all_X_train_input_D, all_X_train_input_P, all_X_val_input_D, all_X_val_input_P, all_X_test_input_D, all_X_test_input_P= [], [], [], [], [], []
    all_Y_train_aff_output, all_Y_val_aff_output, all_Y_test_aff_output= [], [], []
    all_Y_train_output, all_Y_val_output, all_Y_test_output= [], [], []
    
    for v in range(5):
       train_D, train_P, train_I, train_Affi, val_D, val_P, val_I, val_Affi, test_D, test_P, test_I, test_Affi = final_inputs_output(DataName, D, P, Y_Affi, label_row_inds, label_col_inds, train_set, val_set, test_set, v) 

     
       X_train_D = oneHot_OR_catigorical_data(train_D, D_char2int, max_sequence_len)
       X_train_P = oneHot_OR_catigorical_data(train_P, P_char2int, max_sequence_len)
       Y_train = train_I
       Y_train_aff = train_Affi       

       X_val_D = oneHot_OR_catigorical_data(val_D, D_char2int, max_sequence_len)
       X_val_P = oneHot_OR_catigorical_data(val_P, P_char2int, max_sequence_len)
       Y_val = val_I
       Y_val_aff = val_Affi

       X_test_D = oneHot_OR_catigorical_data(test_D, D_char2int, max_sequence_len)
       X_test_P = oneHot_OR_catigorical_data(test_P, P_char2int, max_sequence_len)
       Y_test = test_I
       Y_test_aff = test_Affi

       
       #### Final Input/Output (array)  
       ## input data
       # drugs    
       X_train_input_D = np.asarray(X_train_D)
       X_val_input_D = np.asarray(X_val_D)
       X_test_input_D = np.asarray(X_test_D)
       # protiens
       X_train_input_P = np.asarray(X_train_P)
       X_val_input_P = np.asarray(X_val_P)
       X_test_input_P = np.asarray(X_test_P)
       
       # output data(classification)
       Y_train_output = np.asarray(Y_train) 
       Y_val_output = np.asarray(Y_val) 
       Y_test_output = np.asarray(Y_test) 
       
       # output data(regression)
       Y_train_aff_output = np.asarray(Y_train_aff) 
       Y_train_aff_output =Y_train_aff_output.reshape(-1, 1)
       Y_val_aff_output = np.asarray(Y_val_aff) 
       Y_val_aff_output =Y_val_aff_output.reshape(-1, 1)       
       Y_test_aff_output = np.asarray(Y_test_aff) 
       Y_test_aff_output =Y_test_aff_output.reshape(-1, 1)
       
      
       ##### Final 5_fold CV 
       ## input data
       # drugs    
       all_X_train_input_D.append(X_train_input_D) 
       all_X_val_input_D.append(X_val_input_D) 
       all_X_test_input_D.append(X_test_input_D)
       # protiens
       all_X_train_input_P.append(X_train_input_P) 
       all_X_val_input_P.append(X_val_input_P) 
       all_X_test_input_P.append(X_test_input_P) 
    
       # output data(classification)
       all_Y_train_output.append(Y_train_output) 
       all_Y_val_output.append(Y_val_output)
       all_Y_test_output.append(Y_test_output) 
    
       # output data(regression)
       all_Y_train_aff_output.append(Y_train_aff_output) 
       all_Y_val_aff_output.append(Y_val_aff_output)
       all_Y_test_aff_output.append(Y_test_aff_output)   
            
            
    return all_X_train_input_D, all_X_train_input_P, all_X_val_input_D, all_X_val_input_P, all_X_test_input_D, all_X_test_input_P, all_Y_train_aff_output, all_Y_val_aff_output, all_Y_test_aff_output, all_Y_train_output, all_Y_val_output, all_Y_test_output
    
def final_BindDB_data(DataName, Dtokens, Ptokens, D_char2int, P_char2int, max_sequence_len):
    canSMILE = json.load(open("./data/BindDB_data/canonical_drugs.txt"), object_pairs_hook=OrderedDict)
    
    all_X_train_input_D, all_X_train_input_P, all_X_val_input_D, all_X_val_input_P= [], [], [], []
    all_Y_train_aff_output, all_Y_val_aff_output= [], []
    all_Y_train_output, all_Y_val_output= [], []
    
    for fold in range(5):
        Val_data = pd.read_csv("./data/BindDB_data/DBD_val_v{}.txt".format(fold), delimiter = ",", names = ["ligand_id","smiles","prot_id","aa_sequence","affinity_score"])
        Train_data = pd.read_csv("./data/BindDB_data/DBD_train_v{}.txt".format(fold), delimiter = ",", names = ["ligand_id","smiles","prot_id","aa_sequence","affinity_score"])

        ######### val 
        ######### DATA
        Vligand_id = Val_data["ligand_id"]
        VSMILES = Val_data["smiles"]
        VPROTEINS = Val_data["aa_sequence"]
        Vaffinity_Kd = Val_data["affinity_score"]
        
        # smile to canolical-smile
        for d in range(len(Vligand_id)-1):
            VCANsmi = canSMILE[Vligand_id[d+1]]        
            VSMILES[d+1]=VCANsmi
        # print("validation canSMILES=",VSMILES)
    
        VnewDRUGS=[]
        VnewPROTEINS=[]
        VnewY=[]
        for d in range(len(VSMILES)-1):
            VnewDRUGS.append(VSMILES[d+1]) 
            VnewPROTEINS.append(VPROTEINS[d+1]) 
            VnewY.append(Vaffinity_Kd[d+1])            
        

        ####  list to array 
        val_D = np.stack(VnewDRUGS)
        val_P = np.stack(VnewPROTEINS)
        val_Affi = (np.stack(VnewY)).astype(float)
        val_I = bindDB_Y_binary(val_Affi)
        
        
            
        # encoded label                
        X_val_D = oneHot_OR_catigorical_data(val_D, D_char2int, max_sequence_len)
        X_val_P = oneHot_OR_catigorical_data(val_P, P_char2int, max_sequence_len)
        Y_val_aff = val_Affi
        Y_val = val_I
        # print("---> Y_Val shape:", Y_val.size)
        # print(">>>> num of 1 is:",(Y_val==1).sum() )
        # print(">>>> num of 0 is:",(Y_val==0).sum() )
        
        
        X_val_D = np.asarray(X_val_D) 
        X_val_P = np.asarray(X_val_P) 
        Y_val = np.asarray(Y_val)            
        Y_val_aff = np.asarray(Y_val_aff)
        
        
        ######## train  #######
        ######## DATA   #######
        Tligand_id = Train_data["ligand_id"]
        TSMILES = Train_data["smiles"]
        TPROTEINS = Train_data["aa_sequence"]
        Taffinity_Kd = Train_data["affinity_score"]
        
        for dd in range(len(Tligand_id)-1):
            TCANsmi = canSMILE[Tligand_id[dd+1]]
            TSMILES[dd+1]=TCANsmi
        # print("training canSMILES=",TSMILES)
    
        TnewDRUGS=[]
        TnewPROTEINS=[]
        TnewY=[]
        for d in range(len(TSMILES)-1):
            TnewDRUGS.append(TSMILES[d+1]) 
            TnewPROTEINS.append(TPROTEINS[d+1]) 
            TnewY.append(Taffinity_Kd[d+1]) 

    
        #### from list to array 

        #### from list to array 
        train_D = np.stack(TnewDRUGS)
        train_P = np.stack(TnewPROTEINS)
        train_Affi = (np.stack(TnewY)).astype(float)
        train_I = bindDB_Y_binary(train_Affi)

                            
        
        X_train_D = oneHot_OR_catigorical_data(train_D, D_char2int, max_sequence_len)
        X_train_P = oneHot_OR_catigorical_data(train_P, P_char2int, max_sequence_len)
        Y_train = train_I
        Y_train_aff = train_Affi 
    
        # print("---> Y_train shape:", Y_train.size)
        # print(">>>> num of 1 is:",(Y_train==1).sum() )
        # print(">>>> num of 0 is:",(Y_train==0).sum() )
        
        
        X_train_D = np.asarray(X_train_D)
        X_train_P = np.asarray(X_train_P) 
        Y_train = np.asarray(Y_train)            
        Y_train_aff = np.asarray(Y_train_aff)
        
        
        ##### Final 5_fold CV 
        # input data  
        all_X_train_input_D.append(X_train_D)
        all_X_train_input_P.append(X_train_P)
        all_X_val_input_D.append(X_val_D)
        all_X_val_input_P.append(X_val_P)

        
        # output data(classification)
        all_Y_train_output.append(Y_train)
        all_Y_val_output.append(Y_val)
        
        # output data(regression)
        all_Y_train_aff_output.append(Y_train_aff)
        all_Y_val_aff_output.append(Y_val_aff)
        
    return all_X_train_input_D, all_X_train_input_P, all_X_val_input_D, all_X_val_input_P, all_Y_train_aff_output, all_Y_val_aff_output, all_Y_train_output, all_Y_val_output
   


def FinalProcessingData_for_DavisKiba(DataName):
    max_sequence_len=1100

    if DataName=="Davis":
        max_sequence_len=1100
    else:
        if DataName=="Kiba":
            max_sequence_len=1285
        else:
            print(f"######## {DataName} doesn't exist")

   
    Drugs, Proteins = load_inputData(DataName)
   
    Dtokens, D_char2int= preprocessing_Drugs(Drugs)
    Ptokens, P_char2int= preprocessing_Proteins(Proteins)
    

        
    print("               -----------  FINAL TRAINING/VALIDATION   ----------- ")
    print("               -----------------    DATASETS    -------------------")
    
    X_train_D, X_train_P, X_val_D, X_val_P, X_test_D, X_test_P, Y_train_aff, Y_val_aff, Y_test_aff, Y_train, Y_val, Y_test = final_TrainingValidationTest_data(DataName, Dtokens, Ptokens, D_char2int, P_char2int, max_sequence_len)

    return  X_train_D, X_train_P, X_val_D, X_val_P, Y_train_aff, Y_val_aff, Y_val, D_char2int, P_char2int, max_sequence_len    
    

def FinalProcessingData_for_BindDB(DataName):
    max_sequence_len=1100
    if DataName=="BindDB":
        max_sequence_len=1100
    else:
        print("Error in data name ...")
    
    Drugs, Proteins = load_inputData(DataName)
   
    Dtokens, D_char2int= preprocessing_Drugs(Drugs)
    Ptokens, P_char2int= preprocessing_Proteins(Proteins)
    
    
    
    
    
    print("               -----------  FINAL TRAINING/VALIDATION   ----------- ")
    print("               -----------------    DATASETS    -------------------")
    
    X_train_D, X_train_P, X_val_D, X_val_P, Y_train_aff, Y_val_aff, Y_train, Y_val = final_BindDB_data(DataName, Dtokens, Ptokens, D_char2int, P_char2int, max_sequence_len)
    return  X_train_D, X_train_P, X_val_D, X_val_P, Y_train_aff, Y_val_aff, Y_val, D_char2int, P_char2int, max_sequence_len 
    