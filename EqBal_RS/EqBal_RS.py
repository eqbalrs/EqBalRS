import multiprocessing as mp
import numpy as np
import warnings
import time
import numba
from numba import jit , njit
import optuna

warnings.simplefilter(action='ignore', category=FutureWarning)
dataset = 'movieLens'
model= 'adam'
def init():
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import numpy as np
  import os
  import itertools
  import multiprocessing as mp

  return  train_test_split,pd, np,itertools,mp

@njit(parallel=False)
def testing(user_embed_t,item_embed_t,UserList,ItemList,ratingList,rating_matrix):

    indx = 0 
    pop_err =0
    non_pop_error=0
    cnt_pop = 0
    cnt_non_pop = 0
    error=0 
    cnt_total = 0 
    print("Calculating biases")
    for i,j in zip(UserList,ItemList):
        predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t.T[:,j]),4)
        err = np.round(np.power(ratingList[indx] - predRating,2),4)
        if j in pop_items:
          pop_err += err 
          cnt_pop +=1
        else:
          non_pop_error+= err
          cnt_non_pop+=1
        error = np.round(error+ np.round(err,4),4)
        cnt_total+=1
        indx+=1
    
    pop_bias = ((1/cnt_non_pop)* non_pop_error) - ( (1/cnt_pop)* pop_err )
    sq_error = np.round((np.round(error/cnt_total,7)),4)


    arp,ndcg=calcNDCG_ARP(item_embed_t,user_embed_t,UserList,ItemList,rating_matrix)

 


    return np.round((pop_bias),7),cnt_pop,cnt_non_pop, np.round(pop_err/cnt_pop,7), np.round(non_pop_error/cnt_non_pop,7), sq_error, arp,ndcg


def Squared_Loss_cpu(true_rating, pred_rating):
    prediction_error = (true_rating-pred_rating)**2
    return prediction_error     

@njit(parallel=False)
def trainingAdam(user_embed_t,item_embed_t,m_user,v_user,m_item,v_item,UserList,ItemList,ratingList, latent_factors,wt_items,epochs,lr,beta1,beta2,eps,tune_flag):
    item_embed_t = item_embed_t.T

   
    alpha = lr
    
    for ep in range(epochs):
      print('Epochss: '+str(ep))
      indx  = 0
      for i,j in zip(UserList,ItemList):
        wt_j = np.round(wt_items[j],6)
        #prred= np.round(np.dot(user_embed_t[i,:] , item_embed_t[:,j]),6)
        #m_error+=np.round(ratingList[indx] - prred,6)
        
        for k in range(0,latent_factors):
          pred = np.round(np.dot(user_embed_t[i,:] , item_embed_t[:,j]),6)
          err = np.round(ratingList[indx] - pred,6)
          
          #print('Err is '+str(err))
          middle_term = np.round(-2*wt_j*err,7)
          more_middle_term = np.round(np.round(middle_term,7)*np.round(item_embed_t[k][j],6),7)
          more_middle_term1 = np.round(np.round(middle_term,7)*np.round(user_embed_t[i][k],6),7)
          
          grad_user = more_middle_term #( -2 * wt_j * err * item_embed_t[k][j])
          m_user[i][k] = beta1 * m_user[i][k] + (1.0 - beta1) *grad_user
          v_user[i][k] = beta2 * v_user[i][k] + (1.0 - beta2) *(grad_user**2)

          #m_user[i][k] = np.round(np.round(beta1,6) * np.round(m_user[i][k],7) + np.round((1.0 - beta1),7) * np.round(grad_user,7),7)
          #v_user[i][k] = np.round(np.round(beta2,6) * np.round(v_user[i][k],7) + np.round((1.0 - beta2),7) * np.round((grad_user**2),7),7)
          mhat_user = np.round(np.round(m_user[i][k],7) / np.round((1.0 - beta1**(ep+1)),7),7)
          vhat_user = np.round(np.round(v_user[i][k],7) / np.round((1.0 - beta2**(ep+1)),7),7)
          #print('sub user is '+str(mhat_user / (np.sqrt(vhat_user) + eps)))
          #print('user embed old is '+str(user_embed_t[i][k]))
          user_embed_t[i][k] = user_embed_t[i][k] - lr * mhat_user / (np.sqrt(vhat_user) + eps) 
          
          #Now same for item embedding 
          grad_item = more_middle_term1 #-2 * wt_j * err * user_embed_t[i][k]
          m_item[j][k] = beta1 * m_item[j][k] + (1.0 - beta1) * grad_item
          v_item[j][k] = beta2 * v_item[j][k] + (1.0 - beta2) * (grad_item**2)
          mhat_item = m_item[j][k] / (1.0 - beta1**(ep+1))
          vhat_item = v_item[j][k] / (1.0 - beta2**(ep+1))

          item_embed_t[k][j] = item_embed_t[k][j] - lr * mhat_item / (np.sqrt(vhat_item) + eps) 
        indx +=1 
    m_error=0
    if tune_flag:
        indxx = 0 
        pop_err =0
        non_pop_error=0
        cnt_pop = 0
        cnt_non_pop = 0
        error=0 
        cnt_total = 0 
        for i,j in zip(UserList,ItemList):
        #print(user_embed_t.shape)
        #print(item_embed_t.shape)
            predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t[:,j]),6)
            err = np.round(np.power(ratingList[indxx] - predRating,2),6)
            if j in pop_items:
                pop_err += err 
                cnt_pop +=1
            else:
                non_pop_error+= err
                cnt_non_pop+=1
            error = np.round(error+ np.round(err,4),4)
            cnt_total+=1
            indxx+=1
  
        pop_bias = ((1/cnt_non_pop)* non_pop_error) - ( (1/cnt_pop)* pop_err )
        sq_error = np.round((np.round(error/cnt_total,7)),4)
        m_error=abs(pop_bias) #calcError(item_embed_t.T,user_embed_t,UserList,ItemList,ratingList)
        print(sq_error)
    return m_error,user_embed_t,item_embed_t.T,m_user,v_user,m_item,v_item

@njit(parallel=False)
def training(user_embed_t,item_embed_t,UserList,ItemList,ratingList, latent_factors,wt_items,epochs,lr,tune_flag):
#trained only on traindfr movies and item rest are left as such 

    item_embed_t = item_embed_t.T

    for ep in range(epochs):
        print('Epoch: '+str(ep))
        indx = 0 
        for i,j in zip(UserList,ItemList):
            wt_j = np.round(wt_items[j],6)
                
            for k in range(0,latent_factors):
                pred = np.round(np.dot(user_embed_t[i,:] , item_embed_t[:,j]),4)
                err = np.round(ratingList[indx] - pred,6)
                middle_term = np.round(-2*wt_j*err*lr,7)
                more_middle_term = np.round(np.round(middle_term,7)*np.round(item_embed_t[k][j],3),7)
                more_middle_term1 = np.round(np.round(middle_term,7)*np.round(user_embed_t[i][k],3),7)
                
               
                
                user_embed_t[i][k]= np.round(user_embed_t[i][k] - more_middle_term ,3)
                item_embed_t[k][j]= np.round(item_embed_t[k][j] - more_middle_term1,3)
                
            indx +=1
    m_error=0
    if tune_flag:
        indxx = 0 
        pop_err =0
        non_pop_error=0
        cnt_pop = 0
        cnt_non_pop = 0
        error=0 
        cnt_total = 0 
        for i,j in zip(UserList,ItemList):
        #print(user_embed_t.shape)
        #print(item_embed_t.shape)
            predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t[:,j]),6)
            err = np.round(np.power(ratingList[indxx] - predRating,2),6)
            if j in pop_items:
                pop_err += err 
                cnt_pop +=1
            else:
                non_pop_error+= err
                cnt_non_pop+=1
            error = np.round(error+ np.round(err,4),4)
            cnt_total+=1
            indxx+=1
  
        pop_bias = ((1/cnt_non_pop)* non_pop_error) - ( (1/cnt_pop)* pop_err )
        sq_error = np.round((np.round(error/cnt_total,7)),4)
        m_error=abs(pop_bias) #calcError(item_embed_t.T,user_embed_t,UserList,ItemList,ratingList)
        print(sq_error)

    return m_error,user_embed_t,item_embed_t.T

@njit(parallel=False)
def getMap(n_movies,UserList,ItemList):
    item_user_map = []
    for i in range(n_movies):
      temp = []
      for u,j in zip(UserList,ItemList):
        if j==i:
          temp.append(u)
      item_user_map.append(temp)
    return item_user_map




@njit(parallel=True)
def getTrueRating(item,user,UserList,ItemList,ratingList):
  indx = 0
  for i,j in zip(UserList,ItemList):
    if i==user and j==item:
      return ratingList[indx]
    indx+=1
  return 0

def dual_print(f,*args,**kwargs):
    print(f,*args,**kwargs)
    print(f,*args,**kwargs,file=f)

def pre(f,run,train_test_split,pd, np,itertools,mp):
  dual_print(f,'Started Reading !!')#prepros_movieLens
  dfr = pd.read_csv(dataset+'_full.csv', delimiter=',', header=0,engine='python')
 
  print(dfr)
  
  print(len(np.unique(dfr.MovieID.values)))
  

  train_dfr = pd.read_csv('train_'+str(dataset)+'.csv', delimiter=',',header=0)
  valid_dfr = pd.read_csv('valid_'+str(dataset)+'.csv', delimiter=',',header=0)
  train_dfr = train_dfr.reset_index(drop=True)
  valid_dfr = valid_dfr.reset_index(drop=True)
  

  rating_matrix = dfr.pivot(index='UserID', columns='MovieID', values='Rating')
      #print(rating_matrix)
  n_users, n_movies = rating_matrix.shape#max(dfr['UserID']),max(dfr['MovieID'])#rating_matrix_train.shape
  dual_print(f,'1/n_movies '+str(1/n_movies))
  # Scaling ratings to between 0 and 1, this helps our model by constraining predictions
  min_rating, max_rating = 0,5#dfr['Rating'].min(), dfr['Rating'].max()
 
  #sparcity = rating_matrix.notna().sum().sum() / (n_users * n_movies)
  dual_print(f,"Rating matrix shape "+str(rating_matrix.shape))
  
  rating_matrix[rating_matrix.isna()] =0 # -1



  item_user_map = []
  ##print(rating_matrix)
  for i in range(n_movies):
      indexes= np.where(np.array(rating_matrix.iloc[:,i])!=0) 
      item_user_map.append(indexes)
  #del dfr
  rating_matrix = rating_matrix.values
  dual_print(f,'Done Reading!')
  return dfr,train_dfr,valid_dfr, rating_matrix, item_user_map, n_users, n_movies


  

def preprocessing(f, run,train_test_split,pd, np,itertools,mp):
  dual_print(f,'Started Preprocessing')
  dfr = pd.read_csv('yahoo_full.txt', delimiter='\t', header=None,engine='python')
  #dfr1 = pd.read_csv('yahoo_test.txt', delimiter='\t', header=None,engine='python')
  print(dfr)
  print()
  
  dfr.columns =['UserID', 'MovieID', 'Rating']
  
  dfr['MovieID'] -= 1
  dfr['UserID'] -= 1
  #dfr = dfr[:20000]

  hashmap_userid ={}
  temp = np.unique(dfr.UserID.values)

  for i in range(0,len(temp)):
    hashmap_userid.update({temp[i]:i})
      
  hashmap_movieid ={}
  temp = np.unique(dfr.MovieID.values) 
  for i in range(0,len(temp)):
    hashmap_movieid.update({temp[i]:i})
  
  for index, row in dfr.iterrows():
    dfr.loc[index, 'UserID'] = hashmap_userid[row['UserID']]
    dfr.loc[index, 'MovieID'] = hashmap_movieid[row['MovieID']]

  

  dfr.to_csv('prepros_yahoo_full.csv')
  del hashmap_userid, hashmap_movieid
  
  #train_dfr = dfr
  train_dfr, valid_dfr = train_test_split(dfr, test_size=0.4,shuffle=True)
  
  train_dfr = train_dfr.reset_index(drop=True)
  valid_dfr = valid_dfr.reset_index(drop=True)
  train_dfr.to_csv('train_yahooProcessed.csv')
  valid_dfr.to_csv('valid_yahooProcessed.csv')
  rating_matrix = dfr.pivot(index='UserID', columns='MovieID', values='Rating')
  
  n_users, n_movies = rating_matrix.shape#max(dfr['UserID']),max(dfr['MovieID'])#rating_matrix_train.shape
  dual_print(f,'1/n_movies '+str(1/n_movies))

  # Scaling ratings to between 0 and 1, this helps our model by constraining predictions
  min_rating, max_rating = 0,5#dfr['Rating'].min(), dfr['Rating'].max()
  #rating_matrix = (rating_matrix - min_rating) / (max_rating - min_rating)
  
  #sparcity = rating_matrix.notna().sum().sum() / (n_users * n_movies)
  dual_print(f,"Rating matrix shape "+str(rating_matrix.shape))
  
  rating_matrix[rating_matrix.isna()] =0 # -1

  item_user_map = []

  for i in range(n_movies):
    indexes= np.where(np.array(rating_matrix.iloc[:,i])!=0) 
    item_user_map.append(indexes)
  del dfr
  rating_matrix = rating_matrix.values
  dual_print(f,'Done preprocessing')
  return train_dfr,valid_dfr, rating_matrix, item_user_map, n_users, n_movies


@njit(parallel=False)
def calc_ndcg_score(y_true, y_score):

    y_max_sorted = y_true[y_true.argsort()[::-1]]
    y_true_sorted = y_true[y_score.argsort()[::-1]]

    k = y_true.shape[0]
   

    dcg_score = y_true_sorted[0] - 1
    for i in np.arange(1, k):
        dcg_score += y_true_sorted[i] / np.log2(i + 1)

    max_score =2**(y_max_sorted[0])- 1
    for i in np.arange(1, k):
        max_score += y_max_sorted[i] / np.log2(i + 1)
    
    if max_score!=0:
        ndcg = dcg_score / max_score
    else:
        ndcg = 0 
    return ndcg 
    


@njit(parallel=False)
def calcNDCG_ARP(item_embed_t,user_embed_t,UserList,ItemList,rating_matrix):
    unique_users = np.unique(UserList)
    ndcg_users = []
    arp = 0
    w_count =0 

    cnt = 0 
    for u in unique_users:
        pred_user = []
        
        indx = 0 
        for i,j in zip(UserList,ItemList):
            if i ==u:
                predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t.T[:,j]),4) #eR[i][j]
                pred_user.append(tuple((j,predRating)))
        pred_user=sorted(pred_user, key = lambda t: t[1])
        pred_user.reverse()
        pred_user=pred_user[:10]
        
        true_y=[]
        pred_y=[]
        w_user = 0 
        
        for each in pred_user:
            #each0 is item id and each1 is predicted and u is user id 
            w_i = len(np.where(ItemList==each[0])[0])
            w_user+= w_i
            true_y.append(rating_matrix[u][each[0]])
            pred_y.append(each[1])
        w_count += w_user
        ndcg_score_user =  calc_ndcg_score(np.array(true_y),np.array(pred_y))   #ndcg_score([true_y],[pred_y])
        ndcg_users.append(ndcg_score_user)
        cnt += 1

        indx += 1
    deno = 10*len(UserList)*len(unique_users)
    arp = np.round(w_count /deno,7)
    sum_= 0 
    for each in ndcg_users:
      sum_+= each

    ndcg_final = sum_/cnt


    return arp,ndcg_final


@njit(parallel=False)
def calcDiversity(item_embed_t,user_embed_t,UserList,ItemList):
  print('calculating diveristy')
  unique_users = np.unique(UserList)
  list_of_items_for_users = []
  for u in unique_users:
    pred_user = []
    arp_user= 0
    for i,j in zip(UserList,ItemList):   
      if i==u:
        predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t.T[:,j]),4)
        pred_user.append(tuple((j,predRating)))
    pred_user=sorted(pred_user, key = lambda t: t[1])
    pred_user.reverse()
    pred_user=pred_user[:10]
    temp =[]
    for each in pred_user:
      temp.append(each[0])
    list_of_items_for_users.append(temp)

  #see intersection 
  common = {}
  base  = list_of_items_for_users[0]
  for list_ in list_of_items_for_users[1:]:
    for item  in list_:
      if item in base:
        common.append(item)
  cnt_common = len(common)
  diver = np.round(cnt_common/len(ItemList),7)
  return diver








@njit(parallel=False)
def calcTerms(item_embed_t,user_embed_t,UserList,ItemList,ratingList,wt_items,pop_items):
  indx = 0 
  pop_err =0
  non_pop_error=0
  pop_err_wt =0
  non_pop_error_wt=0
  cnt_pop = 0
  cnt_non_pop = 0
  error=0 
  cnt_total = 0 
  print("Calculating biases")
  for i,j in zip(UserList,ItemList):
      wt_j = np.round(wt_items[j],6)
      predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t.T[:,j]),4)
      err = np.round(np.power(ratingList[indx] - predRating,2),4)
      if j in pop_items:
        pop_err += err 
        pop_err_wt += (wt_j*err) 
        cnt_pop +=1
      else:
        non_pop_error+= err
        non_pop_error+= (wt_j*err)
        cnt_non_pop+=1
      error = np.round(error+ np.round(err,4),4)
      cnt_total+=1
      indx+=1
  
  pop_bias = ((1/cnt_non_pop)* non_pop_error) - ( (1/cnt_pop)* pop_err )
  pop_bias_wt = ((1/cnt_non_pop)* non_pop_error_wt) - ( (1/cnt_pop)* pop_err_wt )
  sq_error = np.round((np.round(error/cnt_total,7)),4)
  return np.round((pop_bias),7), np.round(pop_bias_wt,7),cnt_pop,cnt_non_pop, np.round(pop_err/cnt_pop,7), np.round(non_pop_error/cnt_non_pop,7), sq_error

@njit(parallel=False)
def calcARP(item_embed_t,user_embed_t,UserList,ItemList):
  print('Calculating ARP')
  unique_users = np.unique(UserList)

  arp = 0
  w_count =0 
  for u in unique_users:
    pred_user = []
    arp_user= 0
    for i,j in zip(UserList,ItemList):   
      if i==u:
        predRating = np.round(np.dot(user_embed_t[i,:] , item_embed_t.T[:,j]),4)
        pred_user.append(tuple((j,predRating)))
    pred_user=sorted(pred_user, key = lambda t: t[1])
    pred_user.reverse()
    pred_user=pred_user[:10]
    #print(pred_user)
    w_user = 0 
    for each in pred_user:
      w_i = len(np.where(ItemList==each[0])[0])
      #print(w_i)
      w_user+= w_i
    w_count += w_user

  deno = 10*len(UserList)*len(unique_users)
  arp = np.round(w_count /deno,7)
  #print(arp)
  return arp



def sort_into_pop_nonpop(f, train_dfr,item_user_map,threshold_pop):
  dual_print(f,"started sorting")
  pop_items= []
  non_pop_items =[]
  

  pop_items = pd.read_csv(str(dataset)+'_popular.csv',header=None).values
  non_pop_items = pd.read_csv(str(dataset)+'_non_popular.csv',header=None).values

  pop_items = np.unique(np.array(pop_items))
  non_pop_items = np.unique(np.array(non_pop_items))

  dual_print(f,"Sorted items into {} popular and {} non popular".format(len(pop_items),len(non_pop_items)))
  return pop_items,non_pop_items

train_test_split,pd, np,itertools,mp = init()
  
f = open('./tsndsdnnm.txt', 'a')
dfr,train_dfr,valid_dfr, rating_matrix, item_user_map, n_users, n_movies=pre(f,10, train_test_split,pd, np,itertools,mp)
wt_items = np.array([1/n_movies for i in range(0,n_movies)])
threshold_pop=100
pop_items,non_pop_items=sort_into_pop_nonpop(f, train_dfr,item_user_map,threshold_pop)


def objective(trial):

#beta1 = 0.9 
 #   beta2 = 0.999 
  #  eps = 1e-6 
    latent_factors=20
    user_embed_t = np.round(np.random.uniform(low =0.0 , high = 0.9, size = (n_users,20)),6)  #initilaise acc to complete dataset m_movies and n_users. 
    item_embed_t = np.round(np.random.uniform(low =0.0 , high = 0.9, size = (n_movies,20)),6)
    user_embed_t = np.array(user_embed_t,dtype=np.longdouble)
    item_embed_t = np.array(item_embed_t,dtype=np.longdouble)
    m_user = [ [0.0 for _ in range(0,latent_factors)] for u in train_dfr.UserID.values] 
    v_user = [ [0.0 for _ in range(0,latent_factors)] for u in train_dfr.UserID.values]

    m_item = [ [0.0 for _ in range(0,latent_factors)] for i in train_dfr.MovieID.values] 
    v_item = [ [0.0 for _ in range(0,latent_factors)] for i in train_dfr.MovieID.values]
    m_user = np.array(m_user)
    v_user = np.array(v_user)
    m_item = np.array(m_item)
    v_item = np.array(v_item)
    

    epochs = 40
    alpha = trial.suggest_float('alpha', 0.000001,3)#0.000001, 0.000002)
    beta1 = trial.suggest_float('beta1', 0.8, 1.0)
    beta2 = trial.suggest_float('beta2', 0.899,1.5)
    eps =  trial.suggest_float('eps', 0.00001, 1)
    #m_error,user_embed_t,item_embed_t = training(user_embed_t,item_embed_t,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values, latent_factors,wt_items,epochs,alpha,1)
    m_error,user_embed_t,item_embed_t,m_user,v_user,m_item,v_item= trainingAdam(user_embed_t,item_embed_t,m_user,v_user,m_item,v_item,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values, 20,wt_items,epochs,alpha,beta1,beta2,eps,1)
    print('Adam merror pop bias reduced'+str(m_error))
    return m_error


def main():
  lr =4.75#  .985#2.25# 2.75#4.805067011033194#2.75#0.0000012#0.000001#001#2.75#0.0001 #0.0001 
  beta = 0.00000001 #see below update rule static #0.0000000001#0.00000001   # wt decay parameters 
  train_test_split,pd, np,itertools,mp = init()
  trade_off= 0.01 #0.01
  T = 10 #Timesteps
  f = open('./Output/output_aaai_'+str(model)+'_'+str(dataset)+'_lr_'+str(lr)+'_beta_'+str(beta)+'_tradeoff_'+str(trade_off)+'_timesteps_'+str(T)+'.txt', 'a')

  
  seeds = [0,100,200,300,400,500,600,700,800,900]
  error_over_run=[]
  pop_error_over_run=[]
  non_pop_error_over_run=[]
  pop_bias_over_run=[]
  arp_over_run=[]
  
  threshold_pop = 460    #100
  epochs = 10
  latent_factors=20
  tune_flag=0

  
  

  for run in range(0,10):
    dual_print(f,f,"\n\n################# RUN "+str(run)+" ##################\n" )
    #random.seed(seeds[run])
    np.random.seed(seeds[run])
    dfr,train_dfr,valid_dfr, rating_matrix, item_user_map, n_users, n_movies=pre(f,run, train_test_split,pd, np,itertools,mp)
    dual_print(f,"User Min is "+str(min(train_dfr.UserID.values)))
    dual_print(f,"User Max is "+str(max(train_dfr.UserID.values)))
    dual_print(f,"Movie Min is "+str(min(train_dfr.MovieID.values)))
    dual_print(f,"Movie Max is "+str(max(train_dfr.MovieID.values)))
    dual_print(f,"Unique count of items in validation: "+str(len(np.unique(valid_dfr.MovieID.values))))
    dual_print(f,"Unique count of users in validation: "+str(len(np.unique(valid_dfr.UserID.values))))
    dual_print(f,"Total count validation: "+str(len(valid_dfr)))
    dual_print(f,"Total count training: "+str(len(train_dfr)))
    
    pop_items,non_pop_items=sort_into_pop_nonpop(f, train_dfr,item_user_map,threshold_pop)
    error_over_timestep =[]
    pop_error_over_timestep=[]
    non_pop_error_over_timestep =[]
    pop_bias_over_timestep = []
    arp_over_timestep=[]



   
    #model = MatrixFactorization(n_users, n_movies, latent_factors=20) #Model 

    wt_items = np.array([1/n_movies for i in range(0,n_movies)])
                    #t
    
    print("Number of entries in wt items are "+str(len(wt_items)))

#Initialise embeddings 
    user_embed_t = np.round(np.random.uniform(low =0.0 , high = 0.9, size = (n_users,latent_factors)),6)  #initilaise acc to complete dataset m_movies and n_users. 
    item_embed_t = np.round(np.random.uniform(low =0.0 , high = 0.9, size = (n_movies,latent_factors)),6)
    user_embed_t = np.array(user_embed_t,dtype=np.longdouble)
    item_embed_t = np.array(item_embed_t,dtype=np.longdouble)

    m_user = [ [0.0 for _ in range(0,latent_factors)] for u in dfr.UserID.values] 
    v_user = [ [0.0 for _ in range(0,latent_factors)] for u in dfr.UserID.values]

    m_item = [ [0.0 for _ in range(0,latent_factors)] for i in dfr.MovieID.values] 
    v_item = [ [0.0 for _ in range(0,latent_factors)] for i in dfr.MovieID.values]
    m_user = np.array(m_user)
    v_user = np.array(v_user)
    m_item = np.array(m_item)
    v_item = np.array(v_item)
    
    if tune_flag:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
   # study.optimize(objective, n_trials=100)
    #study.optimize(lambda trial: objective(trial,user_embed_t,item_embed_t,m_user,v_user,m_item,v_item,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values, latent_factors,wt_items,epochs,lr), n_trials=100)

        trial = study.best_trial


        print('Trial_no: {} '.format(trial.value))
        print("Parameters: {}".format(trial.params))
       
    params= {'alpha': 1.1720940460849587, 'beta1': 0.9856903694986798, 'beta2': 1.1706461353375781, 'eps': 0.5335551849723481} #40 epochs  0.0003 pop bias 
    
    if model=='adam':
        beta1= 0.9356#params['beta1']
        lr=4.75#1.985#1.985#params['alpha']
        beta2=1.18#params['beta2']
        eps=0.535#params['eps']
    else:
        lr =2.75# 4.575#4.805067011033194

    for t in range(0,T):
      start_time = time.time()
      dual_print(f,"\n\nTime step : "+str(t))
      if model=='sgd':
          _,user_embed_t,item_embed_t = training(user_embed_t,item_embed_t,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values, latent_factors,wt_items,epochs,lr,tune_flag)
      else:
          _,user_embed_t,item_embed_t,m_user,v_user,m_item,v_item = trainingAdam(user_embed_t,item_embed_t,m_user,v_user,m_item,v_item,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values, latent_factors,wt_items,epochs,lr,beta1,beta2,eps,tune_flag)
      
      pop_bias_unwt, pop_bias,cnt_pop,cnt_non_pop,pop_term_unwt,non_pop_term_unwt,sq_error = calcTerms(item_embed_t,user_embed_t,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values,wt_items,pop_items)
      arp,ndcg = calcNDCG_ARP(item_embed_t,user_embed_t,train_dfr.UserID.values,train_dfr.MovieID.values,rating_matrix)
      #calcARP(item_embed_t,user_embed_t,train_dfr.UserID.values,train_dfr.MovieID.values)

      arp_over_timestep.append(np.round(arp,4))
      pop_bias_over_timestep.append(np.round(pop_bias_unwt,4)) #wt pop bias 
      non_pop_error_over_timestep.append(np.round(non_pop_term_unwt,4))
      pop_error_over_timestep.append(np.round(pop_term_unwt,4))
      print("Popularity bias is "+str(pop_bias_unwt))
      print("Popularity bias wt is "+str(pop_bias))
      print("Popular error "+str(pop_term_unwt))
      print("Non Popular error "+str(non_pop_term_unwt))
      print("Sq error "+str(sq_error))
      print("ARP: "+str(arp))
      print("NDCG: "+str(ndcg))
      total_sq_error =0
      cnt =0
      #Calculate Z derivative 
      cp = 1
      for item in np.unique(train_dfr.MovieID.values):
        
        popular = 0 
        
        #if len(item_user_map[item][0])> threshold_pop:
        if item in pop_items:
            popular = 1
        users_for_item_list = item_user_map[item][0]
        
        loss_users_for_ith_item = 0 
        
        for user in np.unique(users_for_item_list):

          trueRating = rating_matrix[user][item] #getTrueRating(item,user,train_dfr.UserID.values,train_dfr.MovieID.values,train_dfr.Rating.values) #rating_matrix[user][item]
          predRating = np.round(np.dot(user_embed_t[user,:] , item_embed_t.T[:,item]),4)

          sq_loss = np.round((trueRating-predRating)**2,4)
          
          loss_users_for_ith_item =np.round(np.round(loss_users_for_ith_item,4)+ np.round(sq_loss,4),4)
          total_sq_error = np.round(np.round(total_sq_error,4)+ np.round(sq_loss,4),4)
      
        total_derv =0
        mid_term = np.round(2*np.round(trade_off,4) * np.round(pop_bias,7),6)
        if popular:  #here wt bias will come  
            total_derv = np.round( np.round(loss_users_for_ith_item,4) *np.round(( 1 - np.round((mid_term /cnt_pop),6)),6) ,6)     
        else:
            total_derv =  np.round( np.round(loss_users_for_ith_item,4) *np.round(( 1 + np.round((mid_term /cnt_non_pop),6)),6) ,6)
        
        if cp==1:
            print("total_derv is "+str(total_derv))
            cp=0
        #if total_derv > 10000:
         #   print("extra large total derv "+str(total_derv))0.000001
        wt_items[item] = np.round(wt_items[item] - 0.00000001*np.round(total_derv,6),6) #(np.round(beta,4)*np.round(total_derv,4)),4)
        
        cnt +=1
        del loss_users_for_ith_item
       
      total_sq_error = sq_error#np.round(np.sqrt(total_sq_error / len(train_dfr)),4)
      dual_print(f,"Total squared error "+str(total_sq_error))
      error_over_timestep.append(np.round(sq_error,4))
      print("TOtal number of weights update is "+str(cnt))
      #print(wt_items)
      end_time = time.time()
      print("Time taken is "+str(end_time-start_time))
    #T timesteps over 
    arp_over_run.append(np.round(arp,4))
    error_over_run.append(np.round(sq_error,4))
    pop_bias_over_run.append(np.round(pop_bias_unwt,4))
    pop_error_over_run.append(np.round(pop_term_unwt,4))
    non_pop_error_over_run.append(np.round(non_pop_term_unwt,4))
    dual_print(f,"ARP over Timestep "+str(arp))
    dual_print(f,"Error over Timestep "+str(error_over_timestep))
    dual_print(f,"Pop Bias over Timestep "+str(pop_bias_over_timestep))
    dual_print(f,"Pop Error over Timestep "+str(pop_error_over_timestep))
    dual_print(f,"Non pop error over Timestep "+str(non_pop_error_over_timestep))
    dual_print(f,"Starting Testing !!")
    pop_bias_test,cnt_pop,cnt_non_pop,pop_term_test,non_pop_term_test,sq_error_test,arp_test,ndcg_test=testing(user_embed_t,item_embed_t,valid_dfr.UserID.values,valid_dfr.MovieID.values,valid_dfr.Rating.values,rating_matrix)
    dual_print(f,"Testing ")
    dual_print(f,"Pop bias "+str(pop_bias_test))
    dual_print(f,"Pop error "+str(pop_term_test))
    dual_print(f,"non Pop error "+str(non_pop_term_test))
    dual_print(f,"overall error "+str(sq_error_test))
    dual_print(f,"ARP "+str(arp_test))
    dual_print(f,"NDCG "+str(ndcg_test))

    del error_over_timestep,pop_bias_over_timestep,pop_error_over_timestep,non_pop_error_over_timestep,wt_items,arp_over_timestep
  #10runs over 
  dual_print(f,"ARP over run "+str(arp_over_run))
  dual_print(f,"Error over run "+str(error_over_run))
  dual_print(f,"Pop Bias over run "+str(pop_bias_over_run))
  dual_print(f,"Pop term error unwt "+str(pop_error_over_run))
  dual_print(f,"Non pop term error unwt "+str(non_pop_error_over_run))
  dual_print(f,"Mean over runs arp "+str(np.round(np.nanmean(np.array(arp_over_run)),4)))
  dual_print(f,"Std Dev over runs arp "+str(np.round(np.nanstd(np.array(arp_over_run)),4)))
  dual_print(f,"Mean over runs error "+str(np.round(np.nanmean(np.array(error_over_run)),4)))
  dual_print(f,"Std Dev over runs error "+str(np.round(np.nanstd(np.array(error_over_run)),4)))
  dual_print(f,"Mean over runs pop bias "+str(np.round(np.nanmean(np.array(pop_bias_over_run)),4)))
  dual_print(f,"Std Dev over runs pop bias "+str(np.round(np.nanstd(np.array(pop_bias_over_run)),4)))
  dual_print(f,"Mean over runs pop term errror  unwt "+str(np.round(np.nanmean(np.array(pop_error_over_run)),4)))
  dual_print(f,"Std Dev over runs pop term error unwt "+str(np.round(np.nanstd(np.array(pop_error_over_run)),4)))
  dual_print(f,"Mean over runs non pop term error unwt " +str(np.round(np.nanmean(np.array(non_pop_error_over_run)),4)))
  dual_print(f,"Std Dev over runs non pop term error unwt "+str(np.round(np.nanstd(np.array(non_pop_error_over_run)),4)))
'''
'''

if __name__ == '__main__':
 # mp.freeze_support()
  main()
