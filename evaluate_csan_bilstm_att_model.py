from metrics_csan_bilstm_att_model import get_rm2
from utilities import davis_Y_binary, kiba_Y_binary, get_save_paths

from sklearn.metrics import auc, precision_recall_curve 
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt



def evaluate_model(hyperparameter_configuration, model, DataName, X_val_inp_D, X_val_inp_P, Y_val_aff_out, Y_val_out, v):

    MODEL_SAVE_PATH, C_INDEX_PLOT_SAVE_PATH, MSE_PLOT_SAVE_PATH, QQ_PLOT_SAVE_PATH = get_save_paths(hyperparameter_configuration, DataName, v)

    v_loss, v_rmse, v_Cindex = model.evaluate([[X_val_inp_D[v], X_val_inp_P[v]]], Y_val_aff_out[v], batch_size=128, verbose=0)
       

    predicted_aff = model.predict([X_val_inp_D[v], X_val_inp_P[v]])
    print("######## Validation_MSE:  ", (v_loss*100))
    print("######## Validation_Cindex:  ", (v_Cindex*100))


    ## 2. Calculate Rm2
    rm2 = get_rm2(Y_val_aff_out[v], predicted_aff)
    print("######## Rm2 :  ", (rm2[0]*100) )            

    
    ## 3. Calculate AUPR 
    predicted_labels = predicted_aff.copy()
    for pr in range(len(predicted_aff)):
        if (DataName == "Davis" or DataName == "BindDB"):
            predicted_labels[pr] = davis_Y_binary(predicted_aff[pr])
        else:
            if DataName == "Kiba":
                predicted_labels[pr]= kiba_Y_binary(predicted_aff[pr])
            else:
                print("error in binaryzation")
                break

    precision, recall, thres = precision_recall_curve(Y_val_out[v], predicted_labels)
    aupr = auc(recall, precision)
    print("######## AUPR :  ",(aupr*100) ) 
    
    
    ## Draw_QQplot
    x = sm.ProbPlot(predicted_aff)           
    y = sm.ProbPlot(Y_val_aff_out[v])
    qqplot_2samples(x, y, line="45")
    plt.title(f"QQ plot on {DataName} dataset")
    plt.xlabel("Predicted affinity")
    plt.ylabel("Actual affinity")
    plt.savefig(QQ_PLOT_SAVE_PATH)
    plt.show()


    return v_Cindex, v_loss, aupr, rm2

