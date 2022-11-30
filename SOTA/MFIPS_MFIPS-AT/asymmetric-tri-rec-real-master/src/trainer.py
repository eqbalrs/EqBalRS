"""
Codes for training recommenders used in the real-world experiments
in the paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback".
"""
import time
import yaml
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator
from utils.preprocessor import preprocess_datasets
from models.models import MFIPS
import numba
from numba import njit, jit
import pickle


def estimate_pscore(train: np.ndarray, train_mcar: np.ndarray,
                    val: np.ndarray, model_name: str) -> Tuple:
    print("Inside estimate pscrore trainer")
    """Estimate pscore."""
    if 'item' in model_name:
        pscore = np.unique(train[:, 1], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 1]]
        pscore_val = pscore[val[:, 1]]

    elif 'user' in model_name:
        pscore = np.unique(train[:, 0], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 0]]
        pscore_val = pscore[val[:, 0]]

    elif 'both' in model_name:
        user_pscore = np.unique(train[:, 0], return_counts=True)[1]
        user_pscore = user_pscore / user_pscore.max()
        item_pscore = np.unique(train[:, 1], return_counts=True)[1]
        item_pscore = item_pscore / item_pscore.max()
        pscore_train = user_pscore[train[:, 0]] * item_pscore[train[:, 1]]
        pscore_val = user_pscore[val[:, 0]] * item_pscore[val[:, 1]]

    elif 'true' in model_name:
        pscore = np.unique(train[:, 2], return_counts=True)[1] /\
            np.unique(train_mcar[:, 2], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    elif 'nb' in model_name:
        pscore = np.unique(train[:, 2], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    else:  # uniform propensity
        pscore_train = np.ones(train.shape[0])
        pscore_val = np.ones(val.shape[0])

    pscore_train = np.expand_dims(pscore_train, 1)
    pscore_val = np.expand_dims(pscore_val, 1)
    return pscore_train, pscore_val


def train_mfips(sess: tf.Session, model: MFIPS, data: str,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**9,
                model_name: str = 'mf', seed: int = 0,threshold_pop: int=100,tune_flag: int =0) -> Tuple:
    """Train and evaluate the MF-IPS model."""

    print("Inside train mfips ")
    np.random.seed(seed)
    train_loss_list = []
    val_loss_list = []
    
    train_mse_list = []
    val_mse_list = []
    test_mse_list = []
    train_arp_list=[]
    val_arp_list=[]
    test_arp_list=[]
    #val_mae_list =[]
    #test_mae_list = []

    train_pop_bias_list=[]
    val_pop_bias_list=[]
    test_pop_bias_list=[]

    train_pop_term_list=[]
    val_pop_term_list=[]
    test_pop_term_list=[]

    train_non_pop_term_list=[]
    val_non_pop_term_list=[]
    test_non_pop_term_list=[]

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data and estimate the propensity scores.
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=seed)
    pscore_train, pscore_val = estimate_pscore(train=train, train_mcar=train_mcar,
                                               val=val, model_name=model_name)
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)
   
    pop_items_train = []
    non_pop_items_train=[] 
    items_list_train = train[:, 1]
    users_list_train = train[:, 0]
    ratings_train = labels_train

    pop_items_val = []
    non_pop_items_val=[] 
    items_list_val = val[:, 1]
    users_list_val = val[:, 0]
    ratings_val = labels_val
    
    pop_items_test = []
    non_pop_items_test=[] 
    items_list_test = test[:, 1]
    users_list_test = test[:, 0]
    ratings_test = labels_test

    if tune_flag ==0:

        '''user_item_map_train ={}
        for i in np.unique(items_list_train):
            temp=[]
            for u,j in zip(items_list_train,users_list_train):
                if j == i:
                    temp.append(u)
            user_item_map_train.update({i:temp})
        #print(user_item_map_train)
        for i in np.unique(items_list_train):
            if(len(user_item_map_train[i])>threshold_pop):
                pop_items_train.append(i)
            else:
                non_pop_items_train.append(i)

        user_item_map_val ={}
        for i in np.unique(items_list_val):
            temp=[]
            for u,j in zip(items_list_val,users_list_val):
                if j == i:
                    temp.append(u)
            user_item_map_val.update({i:temp})

        for i in np.unique(items_list_val):
            if(len(user_item_map_val[i])>threshold_pop):
                pop_items_val.append(i)
            else:
                non_pop_items_val.append(i)'''

        '''user_item_map_test ={}
        for i in np.unique(items_list_test):
            temp=[]
            for u,j in zip(items_list_test,users_list_test):
                if j == i:
                    temp.append(u)
            user_item_map_test.update({i:temp})

        for i in np.unique(items_list_test):
            if(len(user_item_map_test[i])>threshold_pop):
                pop_items_test.append(i)
            else:
                non_pop_items_test.append(i)'''

        pop_items = pd.read_csv(str(data)+"_popular.csv", header=None)
        pop_items=np.unique(pop_items)
        #print(pop_items)

        non_pop_items = pd.read_csv(str(data)+"_non_popular.csv",header=None)
        
        non_pop_items = np.unique(non_pop_items)
        
        #print(non_pop_items)
        pop_items_train =  []
        non_pop_items_train =[] 
        pop_items_val=[]
        non_pop_items_val=[]
        for i in np.unique(items_list_train):
            if i in pop_items:
                pop_items_train.append(i)
            else:
                non_pop_items_train.append(i)
        
        for i in np.unique(items_list_val):
            if i in pop_items:
                pop_items_val.append(i)
            else:
                non_pop_items_val.append(i)
        
        print("Sorted into {} popular and {} non popular train items".format(len(pop_items_train),len(non_pop_items_train)))
        print("Sorted into {} popular and {} non popular val items".format(len(pop_items_val),len(non_pop_items_val)))
        print('Started DF processing')
        '''pop_items_test=np.unique(pop_items_test)
        non_pop_items_test = np.unique(non_pop_items_test)
        print("Sorted into {} popular and {} non popular test items".format(len(pop_items_test),len(non_pop_items_test)))'''

        df_items_pop_train = []
        df_users_pop_train =[]
        df_rating_pop_train= []
    
        df_items_non_pop_train=[]
        df_users_non_pop_train=[]
        df_rating_non_pop_train=[]
    #
        for idx in range(0,len(items_list_train)):
            if items_list_train[idx] in pop_items_train:
                df_items_pop_train.append( items_list_train[idx] )
                df_users_pop_train.append( users_list_train[idx] )
                df_rating_pop_train.append( ratings_train[idx]   )
            else:
                df_items_non_pop_train.append( items_list_train[idx] )
                df_users_non_pop_train.append( users_list_train[idx] )
                df_rating_non_pop_train.append( ratings_train[idx]   )

        df_items_pop_train = np.array(df_items_pop_train)
        df_users_pop_train = np.array(df_users_pop_train)
        df_rating_pop_train= np.array(df_rating_pop_train)
    
        df_items_non_pop_train= np.array(df_items_non_pop_train)
        df_users_non_pop_train= np.array(df_users_non_pop_train)
        df_rating_non_pop_train= np.array(df_rating_non_pop_train)


        df_items_pop_val = []
        df_users_pop_val =[]
        df_rating_pop_val= []
    
        df_items_non_pop_val=[]
        df_users_non_pop_val=[]
        df_rating_non_pop_val=[]
    
        for idx in range(0,len(items_list_val)):
            if items_list_val[idx] in pop_items_val:
                df_items_pop_val.append( items_list_val[idx] )
                df_users_pop_val.append( users_list_val[idx] )
                df_rating_pop_val.append( ratings_val[idx]   )
            else:
                df_items_non_pop_val.append( items_list_val[idx] )
                df_users_non_pop_val.append( users_list_val[idx] )
                df_rating_non_pop_val.append( ratings_val[idx]   )

        df_items_pop_val = np.array(df_items_pop_val)
        df_users_pop_val = np.array(df_users_pop_val)
        df_rating_pop_val= np.array(df_rating_pop_val)
    
        df_items_non_pop_val= np.array(df_items_non_pop_val)
        df_users_non_pop_val= np.array(df_users_non_pop_val)
        df_rating_non_pop_val= np.array(df_rating_non_pop_val)

        print('Done DF processing')
        '''df_items_pop_test = []
        df_users_pop_test  =[]
        df_rating_pop_test = []
    
        df_items_non_pop_test =[]
        df_users_non_pop_test =[]
        df_rating_non_pop_test =[]
    
        for idx in range(0,len(items_list_test )):
            if items_list_test [idx] in pop_items_test :
                df_items_pop_test .append( items_list_test [idx] )
                df_users_pop_test .append( users_list_test [idx] )
                df_rating_pop_test .append( ratings_test [idx]   )
            else:
                df_items_non_pop_test .append( items_list_test [idx] )
                df_users_non_pop_test .append( users_list_test [idx] )
                df_rating_non_pop_test .append( ratings_test[idx]   )

        df_items_pop_test  = np.array(df_items_pop_test )
        df_users_pop_test  = np.array(df_users_pop_test )
        df_rating_pop_test = np.array(df_rating_pop_test )
    
        df_items_non_pop_test = np.array(df_items_non_pop_test )
        df_users_non_pop_test = np.array(df_users_non_pop_test )
        df_rating_non_pop_test = np.array(df_rating_non_pop_test )'''

    # start running training a recommender
    
    for iter_ in np.arange(max_iters):
        # mini-batch sampling
        idx = np.arange(num_train)   #No batch now          #np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, pscore_batch, labels_batch = train[idx], pscore_train[idx], labels_train[idx]

        # update user-item latent factors
        _, loss = sess.run([model.apply_grads, model.loss],
                           feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                      model.labels: labels_batch, model.scores: pscore_batch})
        train_loss_list.append(loss)
        # calculate validation loss
        val_loss = sess.run(model.weighted_mse,
                            feed_dict={model.users: val[:, 0], model.items: val[:, 1],
                                       model.labels: labels_val, model.scores: pscore_val})
        val_loss_list.append(val_loss)
    #print('Done training, starting calc scores')
        if tune_flag==0:
        # calculate test loss
            '''mse_score_test = sess.run([model.mse], feed_dict={model.users: test[:, 0],model.items: test[:, 1], model.labels: labels_test})
            test_mse_list.append(mse_score_test)
            #test_mae_list.append(mae_score_test)'''
            test_mse_list.append(-1000)
        

            mse_score_val = sess.run([model.mse],feed_dict={model.users: val[:, 0],model.items: val[:, 1],model.labels: labels_val})

            mse_score_train= sess.run([model.mse],feed_dict={model.users: train[:, 0],model.items: train[:, 1],model.labels: labels_train})
        
        
            val_mse_list.append(mse_score_val)
            #val_mae_list.append(mae_score_val)
            train_mse_list.append(mse_score_train)
            #train_mae_list.append(mae_score_train)

            mse_pop_items_train =  sess.run([model.mse],feed_dict={model.users: df_users_pop_train ,model.items: df_items_pop_train ,model.labels: df_rating_pop_train})
            mse_non_pop_items_train = sess.run([model.mse],feed_dict={model.users: df_users_non_pop_train ,model.items: df_items_non_pop_train ,model.labels: df_rating_non_pop_train})
        
            pop_bias_train = np.power(mse_non_pop_items_train,2) - np.power(mse_pop_items_train,2)
            train_pop_bias_list.append(pop_bias_train)
            train_pop_term_list.append(mse_pop_items_train)
            train_non_pop_term_list.append(mse_non_pop_items_train)


            mse_pop_items_val =  sess.run([model.mse],feed_dict={model.users: df_users_pop_val ,model.items: df_items_pop_val ,model.labels: df_rating_pop_val})
            mse_non_pop_items_val = sess.run([model.mse],feed_dict={model.users: df_users_non_pop_val ,model.items: df_items_non_pop_val ,model.labels: df_rating_non_pop_val})
        
            pop_bias_val = np.power(mse_non_pop_items_val,2) - np.power(mse_pop_items_val,2)
            val_pop_bias_list.append(pop_bias_val)
            val_pop_term_list.append(mse_pop_items_val)
            val_non_pop_term_list.append(mse_non_pop_items_val)

            '''mse_pop_items_test =  sess.run([model.mse],feed_dict={model.users: df_users_pop_test ,model.items: df_items_pop_test ,model.labels: df_rating_pop_test})
            mse_non_pop_items_test = sess.run([model.mse],feed_dict={model.users: df_users_non_pop_test ,model.items: df_items_non_pop_test ,model.labels: df_rating_non_pop_test})
        
            pop_bias_test = np.power(mse_non_pop_items_test,2) - np.power(mse_pop_items_test,2)
            test_pop_bias_list.append(pop_bias_test)
            test_pop_term_list.append(mse_pop_items_test)
            test_non_pop_term_list.append(mse_non_pop_items_test)'''
            test_pop_bias_list.append(-1000)
            test_non_pop_term_list.append(-1000)
            test_pop_term_list.append(-1000)
            
    if True:
        if tune_flag ==0:
            #Calculate Train ARP 
            #preds_train= sess.run(model.preds,
             #                         feed_dict={model.users: train[:, 0],
              #                                   model.items: train[:, 1]})
           # preds_train=preds_train
            #print(preds_train)
            print("started train arp mfips")
            '''unique_users_train= np.unique(users_list_train)
            arp_train=0
            w_count_train = 0 
            #print("users unique "+str(unique_users_train))
            for u in unique_users_train:
                indxxx=0
                pred_user_train = []
                for i,j in zip(users_list_train,items_list_train):
                    arp_user_train=0
                    if i!=u:
                        predRating_train = preds_train[indxxx][0]
                        pred_user_train.append(tuple((j,predRating_train)))
                    indxxx+=1
               
                pred_user_train= sorted(pred_user_train,key = lambda t: t[1])
                pred_user_train.reverse()
                pred_user_train = pred_user_train[:10]
                w_user_train = 0 
                for each in pred_user_train:
                    w_i_train=len(np.where(items_list_train==each[0])[0])
                    w_user_train += w_i_train
                
                w_count_train+= w_user_train
            
            deno_train = 10*len(items_list_train)*len(unique_users_train)'''
            arp_train = -100#calcARP(users_list_train,items_list_train,preds_train)#np.round(w_count_train/deno_train,7)
            train_arp_list.append(arp_train)
            print(arp_train)
            
            # Calc ARP Val 
           
            #preds_val= sess.run(model.preds,
             #                         feed_dict={model.users: val[:, 0],
              #                                   model.items: val[:, 1]})
            
            #print(preds_train)
            print("started val arp mfips")
            '''unique_users_val= np.unique(users_list_val)
            arp_val=0
            w_count_val = 0 
            #print("users unique "+str(unique_users_val))
            for u in unique_users_val:
                indxxx=0
                pred_user_val = []
                for i,j in zip(users_list_val,items_list_val):
                    arp_user_val=0
                    if i!=u:
                        predRating_val = preds_val[indxxx][0]
                        pred_user_val.append(tuple((j,predRating_val)))
                    indxxx+=1
               
                pred_user_val= sorted(pred_user_val,key = lambda t: t[1])
                pred_user_val.reverse()
                pred_user_val = pred_user_val[:10]
                w_user_val = 0 
                for each in pred_user_val:
                    w_i_val=len(np.where(items_list_val==each[0])[0])
                    w_user_val += w_i_val
                
                w_count_val+= w_user_val
            
            deno_val = 10*len(items_list_val)*len(unique_users_val)'''
            arp_val = -100#calcARP(users_list_val,items_list_val,preds_val)#np.round(w_count_val/deno_val,7)
            val_arp_list.append(arp_val)
            print(arp_val)

            
            #Calc ARP Test
            '''preds_test= sess.run(model.preds,
                                      feed_dict={model.users: test[:, 0],
                                                 model.items: test[:, 1]})
            
            #print(preds_test)
            print("started test arp mfips")
            unique_users_test= np.unique(users_list_test)
            arp_test=0
            w_count_test = 0 
            #print("users unique "+str(unique_users_test))
            for u in unique_users_test:
                indxxx=0
                pred_user_test = []
                for i,j in zip(users_list_test,items_list_test):
                    arp_user_test=0
                    if i!=u:
                        predRating_test = preds_test[indxxx][0]
                        pred_user_test.append(tuple((j,predRating_test)))
                    indxxx+=1
               
                pred_user_test= sorted(pred_user_test,key = lambda t: t[1])
                pred_user_test.reverse()
                pred_user_test = pred_user_test[:10]
                w_user_test = 0 
                for each in pred_user_test:
                    w_i_test=len(np.where(items_list_test==each[0])[0])
                    w_user_test += w_i_test
                
                w_count_test+= w_user_test
            
            deno_test = 10*len(items_list_test)*len(unique_users_test)
            arp_test = np.round(w_count_test/deno_test,7)
            test_arp_list.append(arp_test)
            print(arp_test) '''


    test_arp_list.append(-1000)
    u_emb, i_emb, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.item_bias])

    sess.close()
    if tune_flag==0:

        return  (np.min(val_loss_list),train_mse_list,val_mse_list,test_mse_list,train_pop_bias_list,val_pop_bias_list ,test_pop_bias_list, train_pop_term_list, val_pop_term_list ,test_pop_term_list,train_non_pop_term_list, val_non_pop_term_list ,test_non_pop_term_list,train_arp_list,val_arp_list,test_arp_list,u_emb,i_emb,i_bias)
    else:
        return np.min(val_loss_list),_

    #return (np.min(val_loss_list),
        ##    test_mse_list[np.argmin(val_loss_list)],
         #   test_mae_list[np.argmin(val_loss_list)],
        #   u_emb, i_emb, i_bias)

@njit(parallel=False)
def calcARP(users_list,items_list,preds):
    unique_users= np.unique(users_list)
    if True:
        if True:
            arp=0
            w_count = 0 
            #print("users unique "+str(unique_users_val))
            for u in unique_users:
                indxxx=0
                pred_user = []
                for i,j in zip(users_list,items_list):
                    arp_user=0
                    if i!=u:
                        predRating = preds[indxxx][0]
                        pred_user.append(tuple((j,predRating)))
                    indxxx+=1
               
                pred_user= sorted(pred_user,key = lambda t: t[1])
                pred_user.reverse()
                pred_user = pred_user[:10]
                w_user = 0 
                for each in pred_user:
                    w_i=len(np.where(items_list==each[0])[0])
                    w_user += w_i
                
                w_count+= w_user
            
            deno = 10*len(items_list)*len(unique_users)
            arp = np.round(w_count/deno,7)
    return arp
            




def train_mfips_with_at(sess: tf.Session, model: MFIPS, mfips1: MFIPS, mfips2: MFIPS, data: str,
                        train: np.ndarray, val: np.ndarray, test: np.ndarray,
                        epsilon: float, pre_iters: int = 500, post_iters: int = 50, post_steps: int = 5,
                        batch_size: int = 2**9, model_name: str = 'naive-at', seed: int = 0,threshold_pop: int=100,tune_flag: int =0) -> Tuple:
    """Train and evaluate the MF-IPS model with asymmetric tri-training."""
    print("Trainign dataset size "+str(len(train)))
    train_loss_list = []
    val_loss_list = []

    train_arp_list=[]
    val_arp_list=[]
    test_arp_list=[]
    
    train_mse_list = []
    val_mse_list = []
    test_mse_list = []
    #val_mae_list =[]
    #test_mae_list = []

    train_pop_bias_list=[]
    val_pop_bias_list=[]
    test_pop_bias_list=[]

    train_pop_term_list=[]
    val_pop_term_list=[]
    test_pop_term_list=[]

    train_non_pop_term_list=[]
    val_non_pop_term_list=[]
    test_non_pop_term_list=[]

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data and estimate the propensity scores.
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=seed)
    pscore_train, pscore_val = estimate_pscore(train=train, train_mcar=train_mcar,
                                               val=val, model_name=model_name)
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)
    pscore_model = np.ones((batch_size, 1))
    


    # Preprocessing popular non-popular 
    pop_items_train = []
    non_pop_items_train=[] 
    items_list_train = train[:, 1]
    users_list_train = train[:, 0]
    ratings_train = labels_train

    pop_items_val = []
    non_pop_items_val=[] 
    items_list_val = val[:, 1]
    users_list_val = val[:, 0]
    ratings_val = labels_val
    
    pop_items_test = []
    non_pop_items_test=[] 
    items_list_test = test[:, 1]
    users_list_test = test[:, 0]
    ratings_test = labels_test
    if tune_flag ==0:

        '''user_item_map_train ={}
        for i in np.unique(items_list_train):
            temp=[]
            for u,j in zip(items_list_train,users_list_train):
                if j == i:
                    temp.append(u)
            user_item_map_train.update({i:temp})
        #print(user_item_map_train)
        for i in np.unique(items_list_train):
            if(len(user_item_map_train[i])>threshold_pop):
                pop_items_train.append(i)
            else:
                non_pop_items_train.append(i)

        user_item_map_val ={}
        for i in np.unique(items_list_val):
            temp=[]
            for u,j in zip(items_list_val,users_list_val):
                if j == i:
                    temp.append(u)
            user_item_map_val.update({i:temp})

        for i in np.unique(items_list_val):
            if(len(user_item_map_val[i])>threshold_pop):
                pop_items_val.append(i)
            else:
                non_pop_items_val.append(i)'''

        '''user_item_map_test ={}
        for i in np.unique(items_list_test):
            temp=[]
            for u,j in zip(items_list_test,users_list_test):
                if j == i:
                    temp.append(u)
            user_item_map_test.update({i:temp})

        for i in np.unique(items_list_test):
            if(len(user_item_map_test[i])>threshold_pop):
                pop_items_test.append(i)
            else:
                non_pop_items_test.append(i)'''

        pop_items = pd.read_csv(str(data)+"_popular.csv", header=None)
        pop_items=np.unique(pop_items)
        #print(pop_items)

        non_pop_items = pd.read_csv(str(data)+"_non_popular.csv",header=None)
        
        non_pop_items = np.unique(non_pop_items)
        
        #print(non_pop_items)
        pop_items_train =  []
        non_pop_items_train =[] 
        pop_items_val=[]
        non_pop_items_val=[]
        for i in np.unique(items_list_train):
            if i in pop_items:
                pop_items_train.append(i)
            else:
                non_pop_items_train.append(i)
        
        for i in np.unique(items_list_val):
            if i in pop_items:
                pop_items_val.append(i)
            else:
                non_pop_items_val.append(i)
        
        print("Sorted into {} popular and {} non popular train items".format(len(pop_items_train),len(non_pop_items_train)))
        print("Sorted into {} popular and {} non popular val items".format(len(pop_items_val),len(non_pop_items_val)))
        print('Started DF processing')

     
        '''pop_items_test=np.unique(pop_items_test)
        non_pop_items_test = np.unique(non_pop_items_test)
        print("Sorted into {} popular and {} non popular test items".format(len(pop_items_test),len(non_pop_items_test)))'''

        df_items_pop_train = []
        df_users_pop_train =[]
        df_rating_pop_train= []
    
        df_items_non_pop_train=[]
        df_users_non_pop_train=[]
        df_rating_non_pop_train=[]
    #
        for idx in range(0,len(items_list_train)):
            if items_list_train[idx] in pop_items_train:
                df_items_pop_train.append( items_list_train[idx] )
                df_users_pop_train.append( users_list_train[idx] )
                df_rating_pop_train.append( ratings_train[idx]   )
            else:
                df_items_non_pop_train.append( items_list_train[idx] )
                df_users_non_pop_train.append( users_list_train[idx] )
                df_rating_non_pop_train.append( ratings_train[idx]   )

        df_items_pop_train = np.array(df_items_pop_train)
        df_users_pop_train = np.array(df_users_pop_train)
        df_rating_pop_train= np.array(df_rating_pop_train)
    
        df_items_non_pop_train= np.array(df_items_non_pop_train)
        df_users_non_pop_train= np.array(df_users_non_pop_train)
        df_rating_non_pop_train= np.array(df_rating_non_pop_train)


        df_items_pop_val = []
        df_users_pop_val =[]
        df_rating_pop_val= []
    
        df_items_non_pop_val=[]
        df_users_non_pop_val=[]
        df_rating_non_pop_val=[]
    
        for idx in range(0,len(items_list_val)):
            if items_list_val[idx] in pop_items_val:
                df_items_pop_val.append( items_list_val[idx] )
                df_users_pop_val.append( users_list_val[idx] )
                df_rating_pop_val.append( ratings_val[idx]   )
            else:
                df_items_non_pop_val.append( items_list_val[idx] )
                df_users_non_pop_val.append( users_list_val[idx] )
                df_rating_non_pop_val.append( ratings_val[idx]   )

        df_items_pop_val = np.array(df_items_pop_val)
        df_users_pop_val = np.array(df_users_pop_val)
        df_rating_pop_val= np.array(df_rating_pop_val)
    
        df_items_non_pop_val= np.array(df_items_non_pop_val)
        df_users_non_pop_val= np.array(df_users_non_pop_val)
        df_rating_non_pop_val= np.array(df_rating_non_pop_val)
        print('Ended DF processing ')

        '''df_items_pop_test = []
        df_users_pop_test  =[]
        df_rating_pop_test = []
    
        df_items_non_pop_test =[]
        df_users_non_pop_test =[]
        df_rating_non_pop_test =[]
    
        for idx in range(0,len(items_list_test )):
            if items_list_test [idx] in pop_items_test :
                df_items_pop_test .append( items_list_test [idx] )
                df_users_pop_test .append( users_list_test [idx] )
                df_rating_pop_test .append( ratings_test [idx]   )
            else:
                df_items_non_pop_test .append( items_list_test [idx] )
                df_users_non_pop_test .append( users_list_test [idx] )
                df_rating_non_pop_test .append( ratings_test[idx]   )

        df_items_pop_test  = np.array(df_items_pop_test )
        df_users_pop_test  = np.array(df_users_pop_test )
        df_rating_pop_test = np.array(df_rating_pop_test )
    
        df_items_non_pop_test = np.array(df_items_non_pop_test )
        df_users_non_pop_test = np.array(df_users_non_pop_test )
        df_rating_non_pop_test = np.array(df_rating_non_pop_test )'''

    print('Stared pre training')
    # start training a recommender
    np.random.seed(seed)
    # start pre-training step
    for i in np.arange(pre_iters):
        # mini-batch sampling
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        idx1 = np.random.choice(np.arange(num_train), size=batch_size)
        idx2 = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, train_batch1, train_batch2 = train[idx], train[idx1], train[idx2]
        labels_batch, labels_batch1, labels_batch2 = labels_train[idx], labels_train[idx1], labels_train[idx2]
        pscore_batch1, pscore_batch2 = pscore_train[idx1], pscore_train[idx2]
        # update user-item latent factors
        _, train_loss = sess.run([model.apply_grads, model.loss],
                                 feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                            model.labels: labels_batch, model.scores: pscore_model})
        _ = sess.run(mfips1.apply_grads,
                     feed_dict={mfips1.users: train_batch1[:, 0], mfips1.items: train_batch1[:, 1],
                                mfips1.labels: labels_batch1, mfips1.scores: pscore_batch1})
        _ = sess.run(mfips2.apply_grads,
                     feed_dict={mfips2.users: train_batch2[:, 0], mfips2.items: train_batch2[:, 1],
                                mfips2.labels: labels_batch2, mfips2.scores: pscore_batch2})
    # start psuedo-labeling and final prediction steps
    all_data = pd.DataFrame(np.zeros((train[:, 0].max() + 1, train[:, 1].max() + 1)))
    all_data = all_data.stack().reset_index().values[:, :2]
    print('Started Post training')
    for k in np.arange(post_iters):
        for j in np.arange(post_steps):
            idx = np.random.choice(np.arange(all_data.shape[0]), size=num_train )#* 5)
            #print('Total Training Size chosen is '+str(len(idx)))
            batch_data = all_data[idx]
            # create psuedo-labeled dataset (i.e., \tilde{\mathcal{D}})
            preds1, preds2 = sess.run([mfips1.preds, mfips2.preds],
                                      feed_dict={mfips1.users: batch_data[:, 0],
                                                 mfips1.items: batch_data[:, 1],
                                                 mfips2.users: batch_data[:, 0],
                                                 mfips2.items: batch_data[:, 1]})
            idx = np.array(np.abs(preds1 - preds2) <= epsilon).flatten()
            target_users, target_items, pseudo_labels = batch_data[idx, 0], batch_data[idx, 1], preds1[idx]
            target_data = np.c_[target_users, target_items, pseudo_labels]
            # store information during the pseudo-labeleing step
            num_target = target_data.shape[0]                   #Jaha p epilson se kam matched pred... so will be less than total train
            # mini-batch sampling for the pseudo-labeleing step
            idx = np.random.choice(np.arange(num_target), size=batch_size)
            idx1 = np.random.choice(np.arange(num_target), size=batch_size)
            idx2 = np.random.choice(np.arange(num_target), size=batch_size)
            train_batch, train_batch1, train_batch2 = target_data[idx], target_data[idx1], target_data[idx2]
            # update user-item latent factors of the final prediction model
            _, train_loss = sess.run([model.apply_grads, model.mse],
                                     feed_dict={model.users: train_batch[:, 0],
                                                model.items: train_batch[:, 1],
                                                model.labels: np.expand_dims(train_batch[:, 2], 1),
                                                model.scores: np.ones((np.int(batch_size), 1))})
            # calculate validation loss during the psuedo-labeleing step
            val_loss = sess.run(model.weighted_mse,
                               feed_dict={model.users: val[:, 0],
                                          model.items: val[:, 1],
                                           model.scores: pscore_val,
                                           model.labels: labels_val})
            # calculate test losses during the psuedo-labeleing step
           # mse_score, mae_score = sess.run([model.mse, model.mae],
            #                                feed_dict={model.users: test[:, 0], model.items: test[:, 1],
             #                                          model.labels: labels_test})
            #train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            #test_mse_list.append(mse_score)
            #test_mae_list.append(mae_score)

            
            # re-update the model parameters of pre-trained models using pseudo-labeled data
            _ = sess.run(mfips1.apply_grads,
                         feed_dict={mfips1.users: train_batch1[:, 0],
                                    mfips1.items: train_batch1[:, 1],
                                    mfips1.labels: np.expand_dims(train_batch1[:, 2], 1),
                                    mfips1.scores: np.ones((batch_size, 1))})
            _ = sess.run(mfips2.apply_grads,
                         feed_dict={mfips2.users: train_batch2[:, 0],
                                    mfips2.items: train_batch2[:, 1],
                                    mfips2.labels: np.expand_dims(train_batch2[:, 2], 1),
                                    mfips2.scores: np.ones((batch_size, 1))})
           # 
           #ab complete dataset p using original label and train dataset se predict label for train loss & pop bias
            
            if tune_flag==0: 
                mse_pop_items_train =  sess.run([model.mse],feed_dict={model.users: df_users_pop_train ,model.items: df_items_pop_train ,model.labels: df_rating_pop_train})
                mse_non_pop_items_train = sess.run([model.mse],feed_dict={model.users: df_users_non_pop_train ,model.items: df_items_non_pop_train ,model.labels: df_rating_non_pop_train})
        
                pop_bias_train = np.power(mse_non_pop_items_train,2) - np.power(mse_pop_items_train,2)
                train_pop_bias_list.append(pop_bias_train)
                train_pop_term_list.append(mse_pop_items_train)
                train_non_pop_term_list.append(mse_non_pop_items_train)


                mse_pop_items_val =  sess.run([model.mse],feed_dict={model.users: df_users_pop_val ,model.items: df_items_pop_val ,model.labels: df_rating_pop_val})
                mse_non_pop_items_val = sess.run([model.mse],feed_dict={model.users: df_users_non_pop_val ,model.items: df_items_non_pop_val ,model.labels: df_rating_non_pop_val})
        
                pop_bias_val = np.power(mse_non_pop_items_val,2) - np.power(mse_pop_items_val,2)
                val_pop_bias_list.append(pop_bias_val)
                val_pop_term_list.append(mse_pop_items_val)
                val_non_pop_term_list.append(mse_non_pop_items_val)

                '''mse_pop_items_test =  sess.run([model.mse],feed_dict={model.users: df_users_pop_test ,model.items: df_items_pop_test ,model.labels: df_rating_pop_test})
                mse_non_pop_items_test = sess.run([model.mse],feed_dict={model.users: df_users_non_pop_test ,model.items: df_items_non_pop_test ,model.labels: df_rating_non_pop_test})
        
                pop_bias_test = np.power(mse_non_pop_items_test,2) - np.power(mse_pop_items_test,2)
                test_pop_bias_list.append(pop_bias_test)
                test_pop_term_list.append(mse_pop_items_test)
                test_non_pop_term_list.append(mse_non_pop_items_test)'''
                test_pop_bias_list.append(-1000)
                test_non_pop_term_list.append(-1000)
                test_pop_term_list.append(-1000)


                mse_score_train= sess.run([model.mse],feed_dict={model.users: train[:, 0],model.items: train[:, 1],model.labels: labels_train})
           #     mse_score_train = (np.power(mse_pop_items_train,2)*len(df_users_pop_train))+(np.power(mse_non_pop_items_train,2)*len(df_users_non_pop_train))/ (len(df_users_pop_train)+len(df_users_non_pop_train))
                mse_score_val= sess.run([model.mse],feed_dict={model.users: val[:, 0],model.items: val[:, 1],model.labels: labels_val})
                #mse_score_val = (np.power(mse_pop_items_val,2)*len(df_users_pop_val))+(np.power(mse_non_pop_items_val,2)*len(df_users_non_pop_val))/ (len(df_users_pop_val)+len(df_users_non_pop_val))
                '''mse_score_test= sess.run([model.mse],feed_dict={model.users: test[:, 0],model.items: test[:, 1],model.labels: labels_test})
               # mse_score_test = (np.power(mse_pop_items_test,2)*len(df_users_pop_test))+(np.power(mse_non_pop_items_test,2)*len(df_users_non_pop_test))/ (len(df_users_pop_test)+len(df_users_non_pop_test))'''

                val_mse_list.append(mse_score_val)
                train_mse_list.append(mse_score_train)
                test_mse_list.append(-1000) #mse_score_test)



    if True:
        if tune_flag ==0:
            #Calculate Train ARP 
            #preds_train= sess.run(model.preds,
             #                         feed_dict={model.users: train[:, 0],
              #                                   model.items: train[:, 1]})
            #preds_train=preds_train
            #print(preds_train)
            print("started train arp mfips-at")
            '''unique_users_train= np.unique(users_list_train)
            arp_train=0
            w_count_train = 0 
            #print("users unique "+str(unique_users_train))
            for u in unique_users_train:
                indxxx=0
                pred_user_train = []
                for i,j in zip(users_list_train,items_list_train):
                    arp_user_train=0
                    if i!=u:
                        predRating_train = preds_train[indxxx][0]
                        pred_user_train.append(tuple((j,predRating_train)))
                    indxxx+=1
               
                pred_user_train= sorted(pred_user_train,key = lambda t: t[1])
                pred_user_train.reverse()
                pred_user_train = pred_user_train[:10]
                w_user_train = 0 
                for each in pred_user_train:
                    w_i_train=len(np.where(items_list_train==each[0])[0])
                    w_user_train += w_i_train
                
                w_count_train+= w_user_train
            
            deno_train = 10*len(items_list_train)*len(unique_users_train)'''
            arp_train = -100#calcARP(users_list_train,items_list_train,preds_train)#np.round(w_count_train/deno_train,7)
            train_arp_list.append(arp_train)
            print(arp_train)
            # Calc ARP Val 
            
            #preds_val= sess.run(model.preds,
             #                         feed_dict={model.users: val[:, 0],
              #                                   model.items: val[:, 1]})
            
            #print(preds_val)
            print("started val arp mfips at")
            '''unique_users_val= np.unique(users_list_val)
            arp_val=0
            w_count_val = 0 
            #print("users unique "+str(unique_users_val))
            for u in unique_users_val:
                indxxx=0
                pred_user_val = []
                for i,j in zip(users_list_val,items_list_val):
                    arp_user_val=0
                    if i!=u:
                        predRating_val = preds_val[indxxx][0]
                        pred_user_val.append(tuple((j,predRating_val)))
                    indxxx+=1
               
                pred_user_val= sorted(pred_user_val,key = lambda t: t[1])
                pred_user_val.reverse()
                pred_user_val = pred_user_val[:10]
                w_user_val = 0 
                for each in pred_user_val:
                    w_i_val=len(np.where(items_list_val==each[0])[0])
                    w_user_val += w_i_val
                
                w_count_val+= w_user_val
            
            deno_val = 10*len(items_list_val)*len(unique_users_val)'''
            arp_val = -100#calcARP(users_list_val,items_list_val,preds_val) #np.round(w_count_val/deno_val,7)
            val_arp_list.append(arp_val)
            print(arp_val)


            #Calc ARP Test
            '''preds_test= sess.run(model.preds,
                                      feed_dict={model.users: test[:, 0],
                                                 model.items: test[:, 1]})
            
            #print(preds_test)
            print("started test arp mfips-at")
            unique_users_test= np.unique(users_list_test)
            arp_test=0
            w_count_test = 0 
            #print("users unique "+str(unique_users_test))
            for u in unique_users_test:
                indxxx=0
                pred_user_test = []
                for i,j in zip(users_list_test,items_list_test):
                    arp_user_test=0
                    if i!=u:
                        predRating_test = preds_test[indxxx][0]
                        pred_user_test.append(tuple((j,predRating_test)))
                    indxxx+=1
               
                pred_user_test= sorted(pred_user_test,key = lambda t: t[1])
                pred_user_test.reverse()
                pred_user_test = pred_user_test[:10]
                w_user_test = 0 
                for each in pred_user_test:
                    w_i_test=len(np.where(items_list_test==each[0])[0])
                    w_user_test += w_i_test
                
                w_count_test+= w_user_test
            
            deno_test = 10*len(items_list_test)*len(unique_users_test)
            arp_test = np.round(w_count_test/deno_test,7)
            test_arp_list.append(arp_test)
            print(arp_test) '''
    test_arp_list.append(-1000)
    # obtain user-item embeddings
    u_emb, i_emb, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.item_bias])

    sess.close()
    if tune_flag==0:

        return  (np.min(val_loss_list),train_mse_list,val_mse_list,test_mse_list,train_pop_bias_list,val_pop_bias_list ,test_pop_bias_list, train_pop_term_list, val_pop_term_list ,test_pop_term_list,train_non_pop_term_list, val_non_pop_term_list ,test_non_pop_term_list,train_arp_list,val_arp_list,test_arp_list,u_emb,i_emb,i_bias)
    else:
        return np.min(val_loss_list),_

    #return (np.min(val_loss_list),
     #       test_mse_list[np.argmin(val_loss_list)],
      #      test_mae_list[np.argmin(val_loss_list)],
       #     u_emb, i_emb, i_bias)


class Objective:

    def __init__(self, data: str, model_name: str = 'mf') -> None:
        """Initialize Class"""
        self.data = data
        self.model_name = model_name

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""
        train, val, test, num_users, num_items =\
            preprocess_datasets(data=self.data, seed=0)

        # sample a set of hyperparameters.
        config = yaml.safe_load(open('../config.yaml', 'r'))
        eta = config['eta']
        max_iters = config['max_iters']
        batch_size = config['batch_size']
        pre_iters = config['pre_iters']
        post_iters = config['post_iters']
        post_steps = config['post_steps']
        threshold_pop= config['threshold_pop']
        dim = trial.suggest_discrete_uniform('dim', 5, 50, 5)
        lam = trial.suggest_loguniform('lam', 1e-6, 1)
        if '-at' in self.model_name:
            epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1)

        ops.reset_default_graph()
        tf.set_random_seed(0)
        sess = tf.Session()
        print('Started Tuning')
        if '-at' not in self.model_name:
            model = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta)
            score,  _ = train_mfips(
                sess, model=model, data=self.data, train=train, val=val, test=test,
                max_iters=max_iters, batch_size=batch_size, model_name=self.model_name,seed=0,threshold_pop=threshold_pop,tune_flag=1)

        else:
            model = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=0)
            model1 = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=1)
            model2 = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=2)
            score, _,  = train_mfips_with_at(
                sess, model=model, mfips1=model1, mfips2=model2, data=self.data,
                train=train, val=val, test=test, epsilon=epsilon,
                pre_iters=pre_iters, post_iters=post_iters, post_steps=post_steps,
                batch_size=batch_size, model_name=self.model_name,seed=0,threshold_pop=threshold_pop,tune_flag=1)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self, data: str, model_name: str) -> None:
        """Initialize Class."""
        self.data = data
        self.model_name = model_name

    def tune(self, n_trials: int = 30) -> None:
        print("Isnide tune in tuner")
        """Hyperparameter Tuning by TPE."""
        path = Path(f'../logs/{self.data}')
        path.mkdir(exist_ok=True, parents=True)
        path_model = Path(f'../logs/{self.data}/{self.model_name}')
        path_model.mkdir(exist_ok=True, parents=True)
        # tune hyperparameters by Optuna
        objective = Objective(data=self.data, model_name=self.model_name)
        study = optuna.create_study(sampler=TPESampler(seed=123))
        study.optimize(objective, n_trials=n_trials)
        # save tuning results
        study.trials_dataframe().to_csv(str(path_model / f'tuning_results.csv'))
        if Path('../hyper_params.yaml').exists():
            pass
        else:
            yaml.dump(dict(yahoo=dict(), coat=dict(), movie=dict(), pantry=dict(), gift_cards=dict() ),   #Add dataset name 
                      open('../hyper_params.yaml', 'w'), default_flow_style=False)
        time.sleep(np.random.rand())
        hyper_params_dict = yaml.safe_load(open('../hyper_params.yaml', 'r'))
        hyper_params_dict[self.data][self.model_name] = study.best_params
        yaml.dump(hyper_params_dict, open('../hyper_params.yaml', 'w'), default_flow_style=False)


class Trainer:

    def __init__(self, data: str, batch_size: int = 10, eta: float = 0.01,
                 max_iters: int = 500, pre_iters: int = 150, post_iters: int = 10,
                 post_steps: int = 10, model_name: str = 'mf',threshold_pop: int = 100) -> None:
        """Initialize class."""
        self.data = data
        hyper_params = yaml.safe_load(open(f'../hyper_params.yaml', 'r'))[data][model_name]
        self.dim = 20  #hyper_params['dim']
        self.lam = hyper_params['lam']
        if '-at' in model_name:
            self.epsilon = hyper_params['epsilon']
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.pre_iters = pre_iters
        self.post_iters = post_iters
        self.post_steps = post_steps
        self.eta = eta
        self.model_name = model_name
        self.threshold_pop = threshold_pop

    def run_simulations(self, num_sims: int = 5) -> None:
        print("Inside run sumulation ")
        """Train mf."""
        results_mse_train = []  #over simulations ie runs 
        results_mse_val = []
        results_mse_test = []
        #results_mae = []
        results_ndcg_test = []
        results_pop_bias_train =[]
        results_pop_bias_val =[]
        results_pop_bias_test =[]
       
        results_pop_term_train=[]
        results_pop_term_val=[]
        results_pop_term_test=[]

        results_non_pop_term_train=[]
        results_non_pop_term_val=[]
        results_non_pop_term_test=[]

        results_arp_train=[]
        results_arp_val=[]
        results_arp_test=[]
        
        mse_over_iter_over_runs_train=[]
        pop_term_over_iter_over_runs_train=[]
        non_pop_term_over_iter_over_runs_train=[]
        bias_over_iter_over_runs_train=[]
        arp_over_iter_over_runs_train=[]

        mse_over_iter_over_runs_val=[]
        pop_term_over_iter_over_runs_val=[]
        non_pop_term_over_iter_over_runs_val=[]
        bias_over_iter_over_runs_val=[]
        arp_over_iter_over_runs_val=[]

        mse_over_iter_over_runs_test=[]
        pop_term_over_iter_over_runs_test=[]
        non_pop_term_over_iter_over_runs_test=[]
        bias_over_iter_over_runs_test=[]
        arp_over_iter_over_runs_test=[]

        # start running simulations
        start = time.time()
        seeds= [0,100,200,300,400,500,600,700,800,900]
        for seedIdx in np.arange(num_sims):
            seed = seeds[seedIdx]
            train, val, test, num_users, num_items =\
                preprocess_datasets(data=self.data, seed=seed)

            ops.reset_default_graph()
            tf.set_random_seed(seed)

            sess = tf.Session()
            if '-at' not in self.model_name:
                model = MFIPS(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta)

                _,train_mse_list,val_mse_list,test_mse_list,  train_pop_bias_list,val_pop_bias_list, test_pop_bias_list, train_pop_term_list, val_pop_term_list ,test_pop_term_list,  train_non_pop_term_list, val_non_pop_term_list ,test_non_pop_term_list,train_arp_list,val_arp_list,test_arp_list,u_emb,i_emb,i_bias = train_mfips(
                    sess, model=model, data=self.data, train=train, val=val, test=test,
                    max_iters=self.max_iters, batch_size=self.batch_size,
                    model_name=self.model_name, seed=seed,threshold_pop=self.threshold_pop,tune_flag=0)
 
              #  _, mse, mae, u_emb, i_emb, i_bias = train_mfips(
                ##    sess, model=model, data=self.data, train=train, val=val, test=test,
                  #  max_iters=self.max_iters, batch_size=self.batch_size,
                  #  model_name=self.model_name, seed=seed)
            else:
                model = MFIPS(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta, num=0)
                model1 = MFIPS(num_users=num_users, num_items=num_items,
                               dim=self.dim, lam=self.lam, eta=self.eta, num=1)
                model2 = MFIPS(num_users=num_users, num_items=num_items,
                               dim=self.dim, lam=self.lam, eta=self.eta, num=2)
                _,train_mse_list,val_mse_list,test_mse_list,  train_pop_bias_list,val_pop_bias_list, test_pop_bias_list, train_pop_term_list, val_pop_term_list ,test_pop_term_list,  train_non_pop_term_list, val_non_pop_term_list ,test_non_pop_term_list,train_arp_list,val_arp_list,test_arp_list,u_emb,i_emb,i_bias  = train_mfips_with_at(
                    sess, model=model, mfips1=model1, mfips2=model2, data=self.data,
                    train=train, val=val, test=test, epsilon=self.epsilon,
                    pre_iters=self.pre_iters, post_iters=self.post_iters, post_steps=self.post_steps,
                    batch_size=self.batch_size, model_name=self.model_name, seed=seed,threshold_pop=self.threshold_pop,tune_flag=0)
            
            print(f'#{seedIdx+1} {self.model_name}: {np.round((time.time() - start) , 2)} minutes')
            mse_over_iter_over_runs_train.append(train_mse_list)
            pop_term_over_iter_over_runs_train.append(train_pop_term_list)
            non_pop_term_over_iter_over_runs_train.append(train_non_pop_term_list)
            bias_over_iter_over_runs_train.append(train_pop_bias_list)
            arp_over_iter_over_runs_train.append(train_arp_list)


            mse_over_iter_over_runs_val.append(val_mse_list)
            pop_term_over_iter_over_runs_val.append(val_pop_term_list)
            non_pop_term_over_iter_over_runs_val.append(val_non_pop_term_list)
            bias_over_iter_over_runs_val.append(val_pop_bias_list)
            arp_over_iter_over_runs_val.append(val_arp_list)

            mse_over_iter_over_runs_test.append(test_mse_list)
            pop_term_over_iter_over_runs_test.append(test_pop_term_list)
            non_pop_term_over_iter_over_runs_test.append(test_non_pop_term_list)
            bias_over_iter_over_runs_test.append(test_pop_bias_list)
            arp_over_iter_over_runs_test.append(test_arp_list)


            #results_mae.append(mae)
            results_mse_train.append(train_mse_list[-1])  #for this run final end epochs 
            results_mse_val.append(val_mse_list[-1])
            results_mse_test.append(test_mse_list[-1])
          
            results_pop_bias_train.append(train_pop_bias_list[-1])
            results_pop_bias_val.append(val_pop_bias_list[-1])
            results_pop_bias_test.append(test_pop_bias_list[-1])

            results_pop_term_train.append(train_pop_term_list[-1])
            results_pop_term_val.append(val_pop_term_list[-1])
            results_pop_term_test.append(test_pop_term_list[-1])

            results_non_pop_term_train.append(train_non_pop_term_list[-1])
            results_non_pop_term_val.append(val_non_pop_term_list[-1])
            results_non_pop_term_test.append(test_non_pop_term_list[-1])

            results_arp_train.append(train_arp_list[-1])
            results_arp_val.append(val_arp_list[-1])
            results_arp_test.append(test_arp_list[-1])

            ndcg = aoa_evaluator(user_embed=u_emb, item_embed=i_emb, item_bias=i_bias, test=test,seed=seed)
            results_ndcg_test.append(ndcg)
            #Append over the epochs results to external file or some list and at last dump 
         
            
        # aggregate and save the final results
        #OVer diff runs , will take mean and std of these csv later 
        result_path = Path(f'../logs/{self.data}/{self.model_name}')
        result_path.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.DataFrame(results_mse_train, columns=['Train MSE']),
                   pd.DataFrame(results_pop_term_train, columns=['Train Pop_MSE']),
                   pd.DataFrame(results_non_pop_term_train, columns=['Train Non_Pop_MSE']),
                   pd.DataFrame(results_pop_bias_train, columns=['Train Pop_bias']),
                   pd.DataFrame(results_arp_train, columns=['Train ARP']),

                   pd.DataFrame(results_mse_val, columns=['Val MSE']),
                   pd.DataFrame(results_pop_term_val, columns=['Val Pop_MSE']),
                   pd.DataFrame(results_non_pop_term_val, columns=['Val Non_Pop_MSE']),
                   pd.DataFrame(results_pop_bias_val, columns=['Val Pop_bias']),
                   pd.DataFrame(results_arp_val, columns=['Val ARP']),

                   pd.DataFrame(results_mse_test, columns=['Test MSE']),
                   pd.DataFrame(results_pop_term_test, columns=['Test Pop_MSE']),
                   pd.DataFrame(results_non_pop_term_test, columns=['Test Non_Pop_MSE']),
                   pd.DataFrame(results_pop_bias_test, columns=['Test Pop_bias']),
                   pd.DataFrame(results_arp_test, columns=['Test ARP']),

                   pd.DataFrame(results_ndcg_test, columns=['Test nDCG@3'])], 1)\
            .to_csv(str(result_path / 'results.csv'))
        
      
        tt = np.array([['mse','pop_mse','non_pop_mse','bias','arp'], mse_over_iter_over_runs_train,pop_term_over_iter_over_runs_train,non_pop_term_over_iter_over_runs_train,bias_over_iter_over_runs_train,arp_over_iter_over_runs_train ])
        np.savetxt('train_iter_run_'+str(self.model_name)+'_gift.csv', tt,fmt='%s')
        tt = np.array([['mse','pop_mse','non_pop_mse','bias','arp'], mse_over_iter_over_runs_val,pop_term_over_iter_over_runs_val,non_pop_term_over_iter_over_runs_val,bias_over_iter_over_runs_val,arp_over_iter_over_runs_val ])
        np.savetxt('val_iter_run_'+str(self.model_name)+'_gift.csv', tt,fmt='%s')
        '''tt = np.array([['mse','pop_mse','non_pop_mse','bias','arp'], mse_over_iter_over_runs_test,pop_term_over_iter_over_runs_test,non_pop_term_over_iter_over_runs_test,bias_over_iter_over_runs_test,arp_over_iter_over_runs_test])
        np.savetxt('test_iter_run_'+str(self.model_name)+'.csv', tt,fmt='%s')'''


         #  pickle.dump(tt,fp)

        #with open(str(result_path / 'val_iter_run.csv') ,'wb') as fp:
          #  tt = [['mse','pop_mse','non_pop_mse','bias'], mse_over_iter_over_runs_val,pop_term_over_iter_over_runs_val,non_pop_term_over_iter_over_runs_val,bias_over_iter_over_runs_val ]
          
          #  pickle.dump(tt,fp)

        #with open(str(result_path / 'test_iter_run.csv') ,'wb') as fp:
           # tt = [['mse','pop_mse','non_pop_mse','bias'], #mse_over_iter_over_runs_test,pop_term_over_iter_over_runs_test,non_pop_term_over_iter_over_runs_test,bias_over_iter_over_runs_test ]
            #pickle.dump(tt,fp)
          
