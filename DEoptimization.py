#        **DE-based hyperparameter optimization framework**

import pandas as pd
import numpy as np
import random 
import sys
import os
import time
import datetime
from tqdm import tqdm 

import io
from keras.models import Sequential, load_model
from keras.layers import SpatialDropout1D, Multiply, Lambda, Permute, RepeatVector, LSTM, Dense, Flatten, Activation, Dropout, Conv1D, Conv2D, Embedding, GlobalMaxPooling1D
from keras.layers import Input, concatenate,Reshape, GlobalMaxPooling2D, MaxPooling1D ,MaxPooling2D, Bidirectional, Convolution1D, BatchNormalization 
from keras.models import Model
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
import json
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint 
from deap import base, creator, tools, algorithms

import multiprocessing
import argparse
import gc; 
#from dataPrep import data
#from dataPrep import rmse
from metrics_csan_bilstm_att_model import rmse, C_index
from data_processing import FinalProcessingData_for_DavisKiba, FinalProcessingData_for_BindDB
np.random.seed(1)
# **Variable initialization**

from csan_bilstm_att_model import model_definition

# dataset name
# nameDATA="Davis"         #["Davis", "Kiba"]
nameDATA="BindDB"





### general vocabular dictionary 
DP_char_set = { "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, "Q": 65, "X": 66}



# **Basic steps of the DE-based CNN-AbiLSTM**



# **1. Initialization step**


def main(checkpoint_name=None):
    
    if checkpoint_name:
        cp = pickle.load(open(checkpoint_name, "rb"))
        pop = cp["population"]
        start_gen = cp["generation"] 
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        params_val= cp["params_values"]
        params_k= cp["params_keys"]
        save_pop= cp["saved_pop"]       
        print("params_val:",params_val )
        print("params_keys:",params_k )
        print("saved_pop:",save_pop)
        for s, indv in enumerate(save_pop): 
            print("--->", s, indv, indv.fitness.values)
            print("----> Best_individual:", real_parameters(individual_to_parameters(indv)))
            


        print("pop:",pop )
        for k, indiv in enumerate(pop): 
            print("--->", k, indiv, indiv.fitness.values)
            print("----> Best_individual:", real_parameters(individual_to_parameters(indiv)))
            
            
       
        

    else:
 
        pop = toolbox.population(n=population_size)
        print("population:" , pop) 
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
        params_val= params_values
        params_k= params_keys
        print("params_val(None):",params_val )
        print("params_key(None):",params_k )
        save_pop=[]
       
        
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    #saved_pop=[]
   
    pop, log, hof, params_val, params_k = DE_algorithm(pop, save_pop, start_gen, toolbox, CR=0.8, F=0.8, ngen=num_generations, 
                                   stats=stats, halloffame=hof, logbook=logbook, verbose=True, params_values= params_val, params_keys= params_k)
    return pop, log, hof, params_val, params_k



# **2. Evaluation and Update steps**
def DE_algorithm(population, saved_pop, start_gen, toolbox, CR, F, ngen, stats, halloffame, logbook, verbose, params_values= None, params_keys= None):

    start_time = datetime.datetime.now()
    pbar = tqdm(total= ngen-start_gen)

    # step2:evaluation initial population 
    if checkpoint_file == None :  
        print("-- Generation %i --" % start_gen)


        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])    
        
        print("Start of evolution")
    
        
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            
        print("--->>  Evaluated %i individuals" % len(population))
        print("--->>  Set of evaluated individuals  :", population)
        
        
    
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        print(logbook.stream)
        print("-pop:", population)
        
        """
        Store the evaluated population and its fitness 
        """
        for c, chrom in enumerate(population):
            saved_pop.append(chrom)
        print("Initial_evaluated_pop", saved_pop)
        
        
        """
        Save initial pop in dictionary--------------------------
        """
        if 0 % 1 == 0:
            cp = dict(population=population, generation=0, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate(), params_values= params_values, params_keys= params_keys, saved_pop= saved_pop)
        
            
            cp_name = f"./results/DE/{nameDATA}_DE_G{0}.pkl"
            pickle.dump(cp, open(cp_name, "wb")) 
            
        pbar.update(1)    
        gc.collect()
 
        
    #### step3: update initiale generation   
    for gen in range(start_gen+1, ngen):
        print("-- Generation %i --" % gen)
        
        # Best individual
        bst = tools.selBest(population, 1)[0]
        print("Best individual", bst)
        
        # Mutation & Crossover
        for k, agent in enumerate(population):
            b,c= toolbox.select(population)         

            y = toolbox.clone(agent)  
            index = random.randrange(len(agent))       
            for i, value in enumerate(agent):
                if i == index or random.random() > CR:       # crossover
                    #(forward) integer --> real
                    BSTi= ((bst[i]*500)/999)-1               
                    Bi= ((b[i]*500)/999)-1      
                    Ci= ((c[i]*500)/999)-1
                  
                    Yi_real = BSTi + F*(Bi-Ci)               # mutation   
                    
                    #(backword) real ---> integer                    
                    Yi_int = int(((Yi_real+1)*999)/500)        

                    if Yi_int in range(len(params_values[i])):
                        y[i] = Yi_int
                    else:
                        y[i] = random.randrange(len(params_values[i]))
                    
           
          
            

            if y in saved_pop:
                for s in range(len(saved_pop)):
                    if (y==toolbox.clone(saved_pop[s])):
                        #print("clone-saved:", toolbox.clone(saved_pop[s]))
                        y.fitness.values = toolbox.clone(saved_pop[s]).fitness.values
                        print("......individual already evaluated")
            else: 
                
                 y.fitness.values = toolbox.evaluate(y)
                 saved_pop.append(y)
                 print("Updated_saved_pop",saved_pop)


                
            #print("(y,agent):", y.fitness , agent.fitness)
            #print("(y,agent)values:", y.fitness.values , agent.fitness.values)
             
            if (y.fitness >= agent.fitness):
                population[k] = y 
                #print("ok")
            
            gc.collect()

        print("---->>  Evaluated %i individuals" % len(population))          
            

        halloffame.update(population)
        record = stats.compile(population) 
        logbook.record(gen=gen, nevals=len(population), **record)
        print(logbook.stream)
            
        if gen % 1 == 0:
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate(), params_values= params_values, params_keys= params_keys, saved_pop=saved_pop)
        
            
            cp_name = f"./results/DE/{nameDATA}_DE_G{gen}.pkl"
            pickle.dump(cp, open(cp_name, "wb"))
            
        pbar.update(1)    
        
        
    pbar.close()
    print("-- End of (successful) evolution --")    
    print("Time:", datetime.datetime.now()- start_time)
 
    gc.collect()

        

    return population, logbook, halloffame, params_values, params_keys

# **a. Evaluation step:**
def evaluateModel(individual):
    #x_train, x_val, y_train , y_val = data()
    # Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen = FinalProcessingData_for_DavisKiba(nameDATA)
    Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen = FinalProcessingData_for_BindDB(nameDATA)
    
    nb_epochs = 1
    gc.collect()
    
    individual_ = individual_to_parameters(individual)
    new_individual = {params_keys[j]:individual_[j] for j in range(len(individual_))}
    
    ci_sum=0
    ci_mean=0
    for v in range(5):
        # ci = Model_CNN_AbiLSTN(nameDATA, new_individual, individual,  Xtrain, Xval, Ytrain_aff, Yval_aff, v, DPchars_set, DPemb_matrix, vocabBYword2vec, maxSequenceLen)
        ci = Model_CSAN_BiLSTM_Att(nameDATA, new_individual, nb_epochs, Xtrain_D[v], Xtrain_P[v], Xval_D[v], Xval_P[v], Ytrain_aff[v], Yval_aff[v], Dchars_set, Pchars_set, maxSequenceLen, v)

        ci_sum= ci_sum+ci
    
    ci_mean= ci_sum/5
    print("C_index average =", ci_mean*100)
    gc.collect()
    
    return ci_mean,   


def Model_CSAN_BiLSTM_Att(nameData, individual, indiv_, Xt_drugs, Xt_protiens, Xv_drugs, Xv_protiens, Yt, Yv, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len, v):
    model = model_definition(individual, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len)
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, C_index]) 
    mc = ModelCheckpoint(f"./results/DE/{nameData}_model_DE{indiv_}--v{v}.h5", save_weights_only=True, monitor='val_C_index', mode='max', save_best_only=True, verbose=1)   
    hist = model.fit([Xt_drugs, Xt_protiens], Yt, 
              epochs=1,
              batch_size=128,
              verbose=1,
              shuffle=True,
              validation_data=([Xv_drugs, Xv_protiens], Yv),
              callbacks=[mc])
    v_loss, v_rmse, v_Cindex = model.evaluate([[Xv_drugs, Xv_protiens]], Yv, batch_size=128, verbose=0)

    print("######### Validation_Cindex:  ", (v_Cindex*100))
    return v_Cindex


# **b.Updated step:**


def mutation(individual):

        mutat_ind = random.choice(range(len(individual))) 
        
        individual[mutat_ind] = random.choice(range(len(params_values[mutat_ind])))

        return individual,
        

def individual_to_parameters(individual):
    parameters = []  
    for i, val in enumerate(individual):
        if params_values[i][val] is not None:
            parameters.append(params_values[i][val])
        else:
            print("error in parameters") 
    return parameters

    
            
def real_parameters(chromosome):
    real_individual = {params_keys[j]:chromosome[j] for j in range(len(chromosome))}
    return  real_individual 



# **Best network**
if __name__ == "__main__":
    
    population_size = 5     
    num_generations = 20   

    # Real parameters
    params_choices = {
                					
						"NCFD": [16, 32, 64, 128],
						"NCFP": [16, 32, 64, 128],
						"CFSD": [2,3,4,6],
						"CFSP": [2,3,4,6],
                        "LSTMdim": [60, 80, 100, 120],
                        "NFCL": [[128], [256, 128], [512, 256], [512, 128]],
                        "DRFCL": [0.1, 0.2, 0.3, 0.4]}
  
    
  

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
                
                
    toolbox = base.Toolbox()
    
    params= []
    params_values= []
    params_keys= []
            
    for key, value in params_choices.items():
                toolbox.register(str(key), random.randint, 0, len(value)-1)
                params.append(getattr(toolbox, str(key)))
                params_values.append(value)
                params_keys.append(key)
         
            
               
    toolbox.register("individual", tools.initCycle, creator.Individual, params, n=1)
            
    toolbox.register("population", tools.initRepeat, list , toolbox.individual)
            
    toolbox.register("select", tools.selRandom, k=2)
    toolbox.register("evaluate", evaluateModel)
    
 
                                               
 
            
    checkpoint_file= None
    # checkpoint_file = "./results/DE/Davis_DE_G1.pkl"
    if checkpoint_file == None :
        
            pop, log, hof, params_val, params_k = main(checkpoint_file) 
  
            
            # Display the best individual 
            best_ind = tools.selBest(pop, 1)[0]
            print("                                 %s  dataset                                     " % (nameDATA)) 
            print(" ===============================     Best individual     ========================== ")
            print("                                                                                      ")    
            print("----> Best_individual (parameters/fitness):  %s, %s" % (individual_to_parameters(best_ind), best_ind.fitness.values))
            print("----> Best_individual (real parameters/fitness):  %s, %s" % (real_parameters(individual_to_parameters(best_ind)), best_ind.fitness.values))
            with open('./results/DE/%s_BestModel.txt' % (nameDATA), 'w') as outfile:
              json.dump(real_parameters(individual_to_parameters(best_ind)), outfile)
          
    else:

            cpp = pickle.load(open(checkpoint_file, "rb")) 
            params_values = cpp["params_values"]
            params_keys = cpp["params_keys"]
            
            pop, log, hof, params_val, params_k = main(checkpoint_file)
 
            
            
            
            # Display the best individual 
            best_ind = tools.selBest(pop, 1)[0]
            
            
            print("                                   %s  dataset                                     " % (nameDATA)) 
            print(" ===================================    Best individual     ========================= ")
            print("                                                                                      ") 
            print("----> Best_individual (parameters/fitness):  %s, %s" % (individual_to_parameters(best_ind), best_ind.fitness.values))
            print("----> Best_individual (real parameters/fitness):  %s, %s" % (real_parameters(individual_to_parameters(best_ind)),  best_ind.fitness.values))
            with open('./results/DE/%s_BestModel.txt' % (nameDATA), 'w') as outfile:
              json.dump(real_parameters(individual_to_parameters(best_ind)), outfile)












