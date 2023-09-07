from data_processing import FinalProcessingData_for_DavisKiba, FinalProcessingData_for_BindDB
from train_evaluate_csan_bilstm_att_model import Train_and_Evaluate

import argparse


DataName = "Kiba"
hyperparameter_configuration = {
                                "NCFD" : 32, 
                                "NCFP" : 64, 
                                "CFSD" : 4, 
                                "CFSP" : 4, 
                                "LSTMdim" : 60,
                                "NFCL" : [512, 256], 
                                "DRFCL" : 0.2
                                }
NB_epochs = 200
MODEL_PATH = "./final_results/SMILES/Evolutionary/DE/Kiba 97.10/Kiba_[32, 64, 4, 60, 512, 256, 0.2]_smiles-v1.h5"


parser = argparse.ArgumentParser(description="Manage variables")
dataset_group = parser.add_mutually_exclusive_group(required=False)
dataset_group.add_argument("-D", "--davis", action="store_true", help="select Davis dataset")
dataset_group.add_argument("-K", "--kiba", action="store_true", help="select Kiba dataset")
dataset_group.add_argument("-B", "--binddb", action="store_true", help="select BindDB dataset")
parser.add_argument("-T", "--train", action="store_true", help="Enable training")

args = parser.parse_args()

if args.train:
    if args.davis:
        DataName = "Davis"
    elif args.kiba:
        DataName = "Kiba"
    elif args.binddb:
        DataName = "BindDB"
    MODEL_PATH = ""
else:
    if args.davis:
        DataName = "Davis"
        hyperparameter_configuration = {
                                "NCFD" : 32, 
                                "NCFP" : 64, 
                                "CFSD" : 3, 
                                "CFSP" : 3, 
                                "LSTMdim" : 60,
                                "NFCL" : [512, 256], 
                                "DRFCL" : 0.2
                                }
        MODEL_PATH = "./final_results/SMILES/Evolutionary/DE/Davis 89.85/Davis_[32, 64, 3, 60, 512, 256, 0.2]-v3.h5"
    elif args.kiba:
        DataName = "Kiba"
        MODEL_PATH = "./final_results/SMILES/Evolutionary/DE/Kiba 97.10/Kiba_[32, 64, 4, 60, 512, 256, 0.2]_smiles-v1.h5"
    elif args.binddb:
        DataName = "BindDB"
        MODEL_PATH = "./final_results/SMILES/Evolutionary/DE/BindDB 84.8/BindDB_(NCFD_32,NCFP_64,CFSD_4,CFSP_4,LSTMdim_60,NFCL_[512, 256],DRFCL_0.2)-v0.h5"



print(">>> Selected Dataset: ", DataName)
if len(MODEL_PATH) == 0:
    print(">>> Training Enabled: True")
else:
    print(">>> No training, only validation of results.")


if DataName == "BindDB":
    Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen = FinalProcessingData_for_BindDB(DataName)
else:
    Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen = FinalProcessingData_for_DavisKiba(DataName)


CI_average, MSE_average, Rm2_average, AUPR_average = Train_and_Evaluate(DataName, MODEL_PATH, hyperparameter_configuration, NB_epochs, Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen)
       


