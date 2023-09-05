import tensorflow as tf
from tensorflow.keras.layers import SpatialDropout1D, Multiply, Lambda, Permute, RepeatVector, LSTM, Dense, Flatten, Activation, Dropout, Embedding, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.layers import Input, Bidirectional, Convolution1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)



def model_definition(hyperparameter_configuration, drugs_vocab_dic, protiens_vocab_dic, max_sequence_len):
    EMBEDDING_DIM = 128

    nbr_filter_D = hyperparameter_configuration["NCFD"]
    nbr_filter_P = hyperparameter_configuration["NCFP"]
    filter_size_D = hyperparameter_configuration["CFSD"]
    filter_size_P = hyperparameter_configuration["CFSP"]
    lstm_dim =  hyperparameter_configuration["LSTMdim"]
    FC_neurons = hyperparameter_configuration["NFCL"]
    num_FClayer= len(FC_neurons)
    drop_fc = hyperparameter_configuration["DRFCL"]

    ### Define the model
    
    ## Drugs 
    inp_D = Input(shape=(max_sequence_len,), dtype='int32') 
    
    # Drug Embeddings
    emb_D = Embedding(input_dim=len(drugs_vocab_dic)+1,
                        output_dim= EMBEDDING_DIM, 
                        input_length=max_sequence_len)(inp_D)
    emb_D = SpatialDropout1D(0.3)(emb_D)
	 
    # CSAN over drugs
    conv_l1_D= Convolution1D(filters=nbr_filter_D, kernel_size= filter_size_D, padding='valid', activation='relu', strides=1)(emb_D)
    conv_l2_D= Convolution1D(filters=nbr_filter_D*2, kernel_size= filter_size_D + 2, padding='valid', activation='relu',strides=1)(conv_l1_D)
    max_pool_l1_D= GlobalMaxPooling1D()(conv_l2_D)
        
    # Self-attention 
    x_a_D = Dense(256, kernel_initializer = 'glorot_uniform', activation="tanh")(max_pool_l1_D)
    x_a_D  = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear')(x_a_D)
    x_a_D = Flatten()(x_a_D)
    att_out_D = Activation('softmax')(x_a_D) # obtain attention weights over Drug Sequence

    x_a2_D = RepeatVector(max_pool_l1_D.shape.as_list()[-1])(att_out_D)
    x_a2_D = Permute([2,1])(x_a2_D)
    mult_D = Multiply()([max_pool_l1_D, x_a2_D]) # attention map for Drugs
    
    att_D = Lambda(lambda x : K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(mult_D) # sum over resulting representaions


    ## Protiens
    inp_P = Input(shape=(max_sequence_len,), dtype='int32') 

    # Protein Embeddings     
    emb_P = Embedding(input_dim=len(protiens_vocab_dic)+1,
                        output_dim= EMBEDDING_DIM, 
                        input_length=max_sequence_len)(inp_P)  
    emb_P = SpatialDropout1D(0.3)(emb_P)
	
    # CSAN over proteins
    conv_l1_P = Convolution1D(filters=nbr_filter_P, kernel_size=filter_size_P, padding='valid', activation='relu', strides=1)(emb_P)
    conv_l2_P = Convolution1D(filters=nbr_filter_P*2, kernel_size=filter_size_P + 2, padding='valid', activation='relu',strides=1)(conv_l1_P)
    max_pool_l1_P= GlobalMaxPooling1D()(conv_l2_P)

    # Self-attention 
    x_a_P = Dense(256, kernel_initializer = 'glorot_uniform', activation="tanh")(max_pool_l1_P) 
    x_a_P  = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear')(x_a_P)
    x_a_P = Flatten()(x_a_P)
    att_out_P = Activation('softmax')(x_a_P) # obtain attention weights over Protein Sequence
    
    x_a2_P = RepeatVector(max_pool_l1_P.shape.as_list()[-1])(att_out_P)
    x_a2_P = Permute([2,1])(x_a2_P)
    mult_P = Multiply()([max_pool_l1_P, x_a2_P]) # attention map for Proteins
    
    att_P = Lambda(lambda x : K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(mult_P)  # sum over resulting representaions


    ## Concatenate the attention maps of Drugs and Proteins
    r_l = Concatenate()([att_D, att_P]) 
    r_l = RepeatVector(1)(r_l)


    ## Attention-based BiLSTM
    layer = Bidirectional(LSTM(lstm_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(r_l)

    # Attention mechanism
    x_a = Dense(lstm_dim*2, kernel_initializer = 'glorot_uniform', activation="tanh")(layer)     
    x_a  = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear')(x_a)
    x_a = Flatten()(x_a)
    att_out = Activation('softmax')(x_a) # obtain attention weights over combined Drug-Proteins sequence
    
    x_a2 = RepeatVector(lstm_dim*2)(att_out)
    x_a2 = Permute([2,1])(x_a2)
    mult = Multiply()([layer, x_a2])
    
    att = Lambda(lambda x : K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(mult) # applying the attention weights to the drug-protein sequence obtained from lstm layer
    	
	
	## Final Dense Layers
    for f in range(num_FClayer):
        att = Dense(FC_neurons[f], activation='relu')(att)
        att = Dropout(drop_fc)(att)
 	
    ## Output Layer
    if type==1:
       out = Dense(1, activation="sigmoid" )(att)
    else:
       out = Dense(1, kernel_initializer='normal')(att)


    ## Build the model
    model = Model(inputs=[inp_D, inp_P], outputs=out)

    return model


