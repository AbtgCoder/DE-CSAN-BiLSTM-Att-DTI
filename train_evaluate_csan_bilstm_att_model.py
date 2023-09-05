from train_csan_bilstm_att_model import train_model, csan_bilstm_att_load_model
from evaluate_csan_bilstm_att_model import evaluate_model



def Train_and_Evaluate(DataName, model_path, hyperparameter_configuration, nb_epochs, X_train_inp_D, X_train_inp_P, X_val_inp_D, X_val_inp_P, Y_train_aff_out, Y_val_aff_out, Y_val_out, D_chars_set, P_chars_set, max_sequence_len):
    
    CI_sum, MSE_sum, Rm2_sum, AUPR_sum = 0, 0, 0, 0
    CI_mean, MSE_mean, Rm2_mean, AUPR_mean = 0, 0, 0, 0  
    for v in range(5):
        if len(model_path) > 0:
            ## Load Model from path
            print("------ Load CSAN-BiLSTM-Att model -----")
            
            model = csan_bilstm_att_load_model(model_path, hyperparameter_configuration, D_chars_set, P_chars_set, max_sequence_len)
        else:
            ## Model Training
            print("------ Train CSAN-BiLSTM-Att model -----")
            model = train_model(hyperparameter_configuration, DataName, nb_epochs, X_train_inp_D[v], X_train_inp_P[v], X_val_inp_D[v], X_val_inp_P[v], Y_train_aff_out[v], Y_val_aff_out[v], D_chars_set, P_chars_set, max_sequence_len, v)

        ## Model Evaluation        
        print("------ Evaluate CSAN-BiLSTM-Att model ----")
        v_Cindex, v_loss, aupr, rm2 = evaluate_model(hyperparameter_configuration, model, DataName, X_val_inp_D, X_val_inp_P, Y_val_aff_out, Y_val_out, v) 
        
        if model_path:
            return v_Cindex, v_loss, rm2, aupr

        ## Sum 
        CI_sum= CI_sum+(v_Cindex*100)
        MSE_sum= MSE_sum+(v_loss*100)
        AUPR_sum= AUPR_sum+(aupr*100)
        Rm2_sum= Rm2_sum+(rm2[0]*100)
        

    ## C_index AVERAGE
    CI_mean= CI_sum/5
    print("C_index AVERAGE =", CI_mean)

    ## MSE AVERAGE
    MSE_mean= MSE_sum/5
    print("MSE AVERAGE =", MSE_mean)

    ## AUPR AVERAGE
    AUPR_mean= AUPR_sum/5
    print("AUPR AVERAGE =", AUPR_mean)

    ## Rm2 AVERAGE
    Rm2_mean= Rm2_sum/5
    print("Rm2 AVERAGE =", Rm2_mean)

    
    return CI_mean, MSE_mean, Rm2_mean, AUPR_mean