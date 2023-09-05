from csan_bilstm_att_model import model_definition
from metrics_csan_bilstm_att_model import rmse, C_index

from utilities import get_save_paths

from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow logging level to only display error messages
import tensorflow as tf
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)  # Set TensorFlow logger level to only display error messages


def csan_bilstm_att_load_model(model_path, hyperparameter_configuration, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len):
    model = model_definition(hyperparameter_configuration, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len)
    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, C_index])
    return model

def train_model(hyperparameter_configuration, dataName, nbr_epochs, Xt_drugs, Xt_protiens, Xv_drugs, Xv_protiens, Yt, Yv, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len, v):

    MODEL_SAVE_PATH, C_INDEX_PLOT_SAVE_PATH, MSE_PLOT_SAVE_PATH, QQ_PLOT_SAVE_PATH = get_save_paths(hyperparameter_configuration, dataName, v)

    model = model_definition(hyperparameter_configuration, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len)

    ## Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, C_index]) 

    mc = ModelCheckpoint(MODEL_SAVE_PATH, save_weights_only=True, monitor='val_C_index', mode='max', save_best_only=True, verbose=1)   

    ## Traning the model

    print("*****start training")

    hist = model.fit([Xt_drugs, Xt_protiens], Yt, 
              epochs= nbr_epochs,
              batch_size=128,
              verbose=1,
              shuffle=True,
              validation_data=([Xv_drugs, Xv_protiens], Yv),
              callbacks=[mc])


    model = model_definition(hyperparameter_configuration, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len)
    model.load_weights(MODEL_SAVE_PATH)          
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, C_index])


    ## C-index
    plt.plot(hist.history['C_index'])
    plt.plot(hist.history['val_C_index'])
    plt.title('C_index values on {} dataset'.format(dataName))
    plt.ylabel('C_index')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(C_INDEX_PLOT_SAVE_PATH)
    plt.show()
    

    ## MSE
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('MSE values on {} dataset'.format(dataName))
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(MSE_PLOT_SAVE_PATH)
    plt.show() 
    
    return model