from data_processing import FinalProcessingData_for_DavisKiba, FinalProcessingData_for_BindDB
from train_evaluate_csan_bilstm_att_model import Train_and_Evaluate


DataName = "Kiba" # Kiba or Davis
# DataName = "BindDB" 

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
MODEL_PATH = "./final_results/SMILES/Evolutionary/DE/Kiba 97.10/Kiba_[32, 64, 4, 60, 512, 256, 0.2]_smiles-v1.h5" # Put Model path to evaluate path
# MODEL_PATH = "./results/BindDB_(NCFD_32,NCFP_64,CFSD_4,CFSP_4,LSTMdim_60,NFCL_[512, 256],DRFCL_0.2)-v0.h5" # CI: 84.80 

Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen = FinalProcessingData_for_DavisKiba(DataName)

# Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen = FinalProcessingData_for_BindDB(DataName)

CI_average, MSE_average, Rm2_average, AUPR_average = Train_and_Evaluate(DataName, MODEL_PATH, hyperparameter_configuration, NB_epochs, Xtrain_D, Xtrain_P, Xval_D, Xval_P, Ytrain_aff, Yval_aff, Yval, Dchars_set, Pchars_set, maxSequenceLen)
       


