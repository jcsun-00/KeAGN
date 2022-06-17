import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import KFold

from config import *
from dataset import Dataset
from util.train import *


def train_and_eval(conf=CONF):
    """Train and evaluate the model based on the given conf.

    Args:
        conf (dict, optional): The configuration the exp based on. Defaults to `CONF`.

    Returns:
        dict: The average `P, R, F` in the CV
    """

    # 获取常用配置
    BATCH_SIZE = conf['batch_size']
    device = torch.device(f'cuda:{conf["cuda_index"]}')

    # 打印配置信息
    show_config()
    # 设定随机化种子
    setup_seed(conf['model_seed'])

    ##########################################################
    # 加载数据, 划分验证集与训练集, 并在训练集中划分交叉验证topic
    ##########################################################
    with open(conf['data_path'], 'rb') as data_file:
        data = pickle.load(data_file)
    
    # 初始化KFold
    kf = KFold(10, shuffle=True, random_state=conf['model_seed'])
    # 设定用于记录评估数据的词典
    cv_eval_data = defaultdict(list)
    
    # 进行10折交叉验证
    for fold, index in enumerate(kf.split(data)):
        print('{:=^75}'.format(f' Fold:{fold+1} (train) '))
        show_time()
        show_pid()
        
        train_set, test_set = train_test_split(data, index[0])
        train_set = neg_sampling(train_set, conf['neg_sam_rate'])
        train_ds, test_ds = [Dataset(BATCH_SIZE, x) for x in [train_set, test_set]]
        model = get_model(conf['model_name'], device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, conf['learning_rate'])
        print(f'train_size: {len(train_ds)}, test_size: {len(test_ds)}')
        print(f'model: {type(model).__name__}')
        if hasattr(model, 'MLP_dim_list'): print(f'MLP_dims: {model.MLP_dim_list}')

        
        ##########################################################
        # 模型训练，验证集选择参数，模型评估
        ##########################################################
        model.train()
        for epoch in range(conf['train_epoch']):
            for batch in train_ds.reader(device, True):
                pred = model_forward(model, batch)
                loss = loss_fn(pred, batch[8])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 模型评估
        model.eval()
        with torch.no_grad():
            predicted_all, gold_all = [], []

            for batch in test_ds.reader(device, True):
                pred = model_forward(model, batch)
                predicted = torch.argmax(pred, -1).cpu().numpy()
                predicted_all = np.concatenate([predicted_all, predicted])
                gold = batch[8].cpu().numpy()
                gold_all = np.concatenate([gold_all, gold])

            print('{:=^75}'.format(f' Fold:{fold+1} (test) '))
            show_time()
            
            prf = compute_f1(gold_all, predicted_all)
            for k, v in prf.items():
                print('{}: {:.6f}'.format(k, v))
                cv_eval_data[k].append(v)
            

    # 计算交叉验证获取的p, r, f的平均值
    print('{:=^75}'.format(f' Average '))
    show_time()
    for k, v in cv_eval_data.items():
        print('{}: {:.6f}'.format(k, np.mean(v)))

    res = { }
    for k, v in cv_eval_data.items():
        res[k] = np.mean(v)
    return res

if __name__ == '__main__':
    train_and_eval()






    



    
    

    
