"""Tools for summarizing experimental results."""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.preprocessor import preprocess_datasets


datasets = ['yahoo', 'movie', 'gift_cards']
#metrics = ['train mse (w/o at)','val mse (w/o at)', 'train mse (w/ at)', 'pop mse (w/o at)', 'pop mse (w/ at)', 'non pop mse (w/o at)', 'non pop mse (w/ at)', 'pop bias (w/o at)', 'pop bias (w/ at)' , 'arp (w/o at)', 'arp (w/ at)']
metrics= ['train mse (w/o at)','val mse (w/o at)', 'train pop_mse (w/o at)','val pop_mse (w/o at)','train non_pop_mse (w/o at)','val non_pop_mse (w/o at)', 'train pop_bias (w/o at)','val pop_bias (w/o at)','train arp (w/o at)','val arp (w/o at)',  'train mse (w/ at)','val mse (w at)', 'train pop_mse (w at)','val pop_mse (w at)','train non_pop_mse (w/ at)','val non_pop_mse (w/ at)', 'train pop_bias (w/ at)','val pop_bias (w/ at)','train arp (w/ at)','val arp (w/ at)']
#model_names = ['uniform', 'uniform-at', 'user', 'user-at', 'item', 'item-at', 

 #              'both', 'both-at', 'nb', 'nb-at', 'nb_true', 'nb_true-at']
stats_idx = ['#User', '#Item', '#Rating', 'Sparsity',
             'Avg. rating of train', 'Avg. rating of test', 'KL div']


def calc_kl_div(train: np.ndarray, test: np.ndarray) -> float:
    print("kl diver")
    """Estimate KL divergence of rating distributions between training and test sets."""
    p = np.unique(train[:, 2], return_counts=True)[1] / \
        np.unique(train[:, 2], return_counts=True)[1].sum()
    q = np.unique(test[:, 2], return_counts=True)[1] / \
       np.unique(test[:, 2], return_counts=True)[1].sum()
    return np.round(np.sum(np.where(p != 0, p * np.log(p / q), 0)), 4)


def summarize_data_statistics() -> None:
    print("summary stats")
    """Save dataset statistics with Tex Table Format."""
    stat_data_list = []
    Path('../paper_results').mkdir(exist_ok=True)
    for data in datasets:
        train, _, test, num_users, num_items = preprocess_datasets(data=data)
        num_data = train.shape[0]
        spasity = f'{100 * (num_data / (num_users * num_items)).round(4)}%'
        avg_train, avg_test = train[:, 2].mean().round(3), test[:, 2].mean().round(3)
        kl = calc_kl_div(train, test)
        stat_data = DataFrame(data=[num_users, num_items, num_data, spasity, avg_train, avg_test, kl],
                              index=stats_idx, columns=[data]).T
        stat_data_list.append(stat_data)
    pd.concat(stat_data_list).to_csv('../paper_results/data_stat.csv', sep='&')


#model_names = ['uniform', 'uniform-at', 'user', 'user-at', 'item', 'item-at',
 #              'both', 'both-at', 'nb', 'nb-at', 'nb_true', 'nb_true-at']

model_names = [ 'nb_true','nb_true-at']

def summarize_experimental_results(data: str) -> None:
    print("summary experi")
    """Summarize results with Tex Table format."""
    raw_results_path = Path(f'../logs/{data}')
    paper_results_path = Path(f'../paper_results/{data}')
    paper_results_path.mkdir(exist_ok=True, parents=True)
    results_train_mse_dict = {}
    results_val_mse_dict = {}
    results_train_pop_mse_dict = {}
    results_val_pop_mse_dict = {}
    results_train_non_pop_mse_dict = {}
    results_val_non_pop_mse_dict = {}
    results_train_pop_bias_dict = {}
    results_val_pop_bias_dict = {}
    results_train_arp_dict = {}
    results_val_arp_dict = {}
    #results_ndcg_dict = {}

    results_train_mse_dict_at = {}
    results_val_mse_dict_at = {}
    results_train_pop_mse_dict_at = {}
    results_val_pop_mse_dict_at = {}
    results_train_non_pop_mse_dict_at = {}
    results_val_non_pop_mse_dict_at = {}
    results_train_pop_bias_dict_at = {}
    results_val_pop_bias_dict_at = {}
    results_train_arp_dict_at = {}
    results_val_arp_dict_at = {}
    #results_ndcg_dict_at = {}

    for model_name in model_names:
        results_ = pd.read_csv(str(raw_results_path / f'{model_name}/results.csv'), index_col=0)
        if '-at' in model_name:
            #results_mse_dict_at[model_name[:-3]] = results_['MSE']
            #results_mae_dict_at[model_name[:-3]] = results_['MAE']
            #results_ndcg_dict_at[model_name[:-3]] = results_['nDCG@3']
            
            results_train_mse_dict_at[model_name[:-3]] = results_['Train MSE']
            results_val_mse_dict_at[model_name[:-3]] = results_['Val MSE']
            results_train_pop_mse_dict_at[model_name[:-3]] = results_['Train Pop_MSE']
            results_val_pop_mse_dict_at[model_name[:-3]] = results_['Val Pop_MSE']
            results_train_non_pop_mse_dict_at[model_name[:-3]] = results_['Train Non_Pop_MSE']
            results_val_non_pop_mse_dict_at[model_name[:-3]] = results_['Val Non_Pop_MSE']
            results_train_pop_bias_dict_at[model_name[:-3]] = results_['Train Pop_bias']
            results_val_pop_bias_dict_at[model_name[:-3]] = results_['Val Pop_bias']
            results_train_arp_dict_at[model_name[:-3]] = results_['Train ARP']
            results_val_arp_dict_at[model_name[:-3]] = results_['Val ARP']
            #results_ndcg_dict_at[model_name[:-3]] = results_['nDCG@3']

        else:  

            results_train_mse_dict[model_name] = results_['Train MSE']
            results_val_mse_dict[model_name] = results_['Val MSE']
            results_train_pop_mse_dict[model_name] = results_['Train Pop_MSE']
            results_val_pop_mse_dict[model_name] = results_['Val Pop_MSE']
            results_train_non_pop_mse_dict[model_name] = results_['Train Non_Pop_MSE']
            results_val_non_pop_mse_dict[model_name] = results_['Val Non_Pop_MSE']
            results_train_pop_bias_dict[model_name] = results_['Train Pop_bias']
            results_val_pop_bias_dict[model_name] = results_['Val Pop_bias']
            results_train_arp_dict[model_name] = results_['Train ARP']
            results_val_arp_dict[model_name] = results_['Val ARP']
            #results_ndcg_dict[model_name] = results_['nDCG@3']


    #results_mae = DataFrame(results_mae_dict).describe().round(5).T
    #results_mse = DataFrame(results_mse_dict).describe().round(5).T
    #results_ndcg = DataFrame(results_ndcg_dict).describe().round(5).T
    #results_mae_at = DataFrame(results_mae_dict_at).describe().round(5).T
    #results_mse_at = DataFrame(results_mse_dict_at).describe().round(5).T
    #results_ndcg_at = DataFrame(results_ndcg_dict_at).describe().round(5).T

    results_train_mse = DataFrame(results_train_mse_dict).describe().round(5).T
    results_val_mse = DataFrame(results_val_mse_dict).describe().round(5).T
    results_train_pop_mse = DataFrame(results_train_pop_mse_dict).describe().round(5).T
    results_val_pop_mse = DataFrame(results_val_pop_mse_dict).describe().round(5).T
    results_train_non_pop_mse = DataFrame(results_train_non_pop_mse_dict).describe().round(5).T
    results_val_non_pop_mse = DataFrame(results_val_non_pop_mse_dict).describe().round(5).T
    results_train_pop_bias = DataFrame(results_train_pop_bias_dict).describe().round(5).T
    results_val_pop_bias = DataFrame(results_val_pop_bias_dict).describe().round(5).T
    results_train_arp = DataFrame(results_train_arp_dict).describe().round(5).T
    #results_ndcg = DataFrame(results_ndcg_dict).describe().round(5).T

    results_train_mse_at = DataFrame(results_train_mse_dict_at).describe().round(5).T
    results_val_mse_at = DataFrame(results_val_mse_dict_at).describe().round(5).T
    results_train_pop_mse_at = DataFrame(results_train_pop_mse_dict_at).describe().round(5).T
    results_val_pop_mse_at = DataFrame(results_val_pop_mse_dict_at).describe().round(5).T
    results_train_non_pop_mse_at = DataFrame(results_train_non_pop_mse_dict_at).describe().round(5).T
    results_val_non_pop_mse_at = DataFrame(results_val_non_pop_mse_dict_at).describe().round(5).T
    results_train_pop_bias_at = DataFrame(results_train_pop_bias_dict_at).describe().round(5).T
    results_val_pop_bias_at = DataFrame(results_val_pop_bias_dict_at).describe().round(5).T
    results_train_arp_at = DataFrame(results_train_arp_dict_at).describe().round(5).T
    #results_ndcg_at = DataFrame(results_ndcg_dict_at).describe().round(5).T

    results_list = [results_train_mse, results_val_mse, results_train_pop_mse, results_val_pop_mse,results_train_non_pop_mse, results_val_non_pop_mse,results_train_pop_bias, results_val_pop_bias, results_train_mse_at, results_val_mse_at, results_train_pop_mse_at, results_val_pop_mse_at,results_train_non_pop_mse_at, results_val_non_pop_mse_at,results_train_pop_bias_at, results_val_pop_bias_at]

    #results_list = [results_mae,  results_mse, results_ndcg]
#Train MSE,Train Pop_MSE,Train Non_Pop_MSE,Train Pop_bias,Train ARP,Val MSE,Val Pop_MSE,Val Non_Pop_MSE,Val Pop_bias,Val ARP,Test MSE,Test Pop_MSE,Test Non_Pop_MSE,Test Pop_bias,Test ARP,Test nDCG@3
    results_dict_mean = {}
    results_dict_std ={}
    #print(results_list)
    for results, metric in zip(results_list, metrics):
        results_dict_mean[metric] = results['mean']
        results_dict_std[metric] = results['std']

    DataFrame(results_dict_mean).to_csv(str(paper_results_path / 'overall_results_mean.csv'))
    DataFrame(results_dict_std).to_csv(str(paper_results_path / 'overall_results_std.csv'))
