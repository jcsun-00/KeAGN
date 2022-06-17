import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

from config import *
from dataset import Dataset
from util.train import *

ALL_TOPIC = [
    '1', '3', '4', '5', '7', 
    '8', '12', '13', '14', '16',
    '18', '19', '20', '22', '23', 
    '24', '30', '32', '33', '35',
    '37', '41'
]

def train_and_eval(conf=CONF):
    """Train and evaluate the model based on the given conf.

    Args:
        conf (dict, optional): The configuration the exp based on. Defaults to `CONF`.

    Returns:
        dict: The average `P, R, F` in the 5-Fold CV
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

    # 划分验证集（最后2个topic）
    cv_set, dev_set = topic_split(data, ['37', '41'])
    dev_ds = Dataset(BATCH_SIZE, dev_set)
    # 划分测试集使用的topic
    cv_test_topic = np.split(np.array(ALL_TOPIC[:20]), 5)
    cv_eval_data = defaultdict(list)
    cv_pred_res = dict()
    
    # 将后2个topic作验证集，在前20个topic里进行交叉验证
    for fold, test_topic in enumerate(cv_test_topic):
        print('{:=^75}'.format(f' Fold:{fold+1} (train) '))
        show_time()
        show_pid()

        train_set, test_set = topic_split(cv_set, test_topic)
        train_set = neg_sampling(train_set, conf['neg_sam_rate'])
        train_ds, test_ds = [Dataset(BATCH_SIZE, x) for x in [train_set, test_set]]
        model = get_model(conf['model_name'], device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, conf['learning_rate'])
        # scheduler = ExponentialLR(optimizer, 0.9)

        print(f'test_topic: {test_topic}')
        print(f'train_size: {len(train_ds)}, test_size: {len(test_ds)}, dev_size: {len(dev_ds)}')
        print(f'model: {type(model).__name__}')
        if hasattr(model, 'MLP_dim_list'): print(f'MLP_dims: {model.MLP_dim_list}')

        
        ##########################################################
        # 模型训练，验证集选择参数，模型评估
        ##########################################################
        for epoch in range(conf['train_epoch']):
            model.train()
            for batch in train_ds.reader(device, True):
                pred = model_forward(model, batch)
                loss = loss_fn(pred, batch[8])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # scheduler.step()

            # # 保存模型训练N个epoch后在验证集上的表现（用于选择train_epoch）
            # model.eval()
            # with torch.no_grad():
            #     predicted_all, gold_all = [], []
            #     for batch in dev_ds.reader(device, True):
            #         pred = model_forward(model, batch)
            #         predicted = torch.argmax(pred, -1).cpu().numpy()
            #         predicted_all = np.concatenate([predicted_all, predicted])
            #         gold = batch[8].cpu().numpy()
            #         gold_all = np.concatenate([gold_all, gold])
            #     prf = compute_f1(gold_all, predicted_all)

            #     log_path = f'log/模型调参/pid{os.getpid()}.csv'
            #     with open(log_path, 'a') as f:
            #         f.write('{}, {}, {}, {}, {:.6f}, {:.6f}, {:.6f}\n'.format(
            #             conf['model_seed'], conf['data_seed'], fold,
            #             epoch+1, prf['precision'], prf['recall'], prf['f1_score']))

        # 模型评估
        model.eval()
        with torch.no_grad():
            predicted_all, gold_all = [], []
            ds = dev_ds if CONF['test_set']=='dev' else test_ds

            for batch, source_batch in ds.reader_with_source_data(device, True):
                pred = model_forward(model, batch)
                predicted = torch.argmax(pred, -1).cpu().numpy()
                predicted_all = np.concatenate([predicted_all, predicted])
                gold = batch[8].cpu().numpy()
                gold_all = np.concatenate([gold_all, gold])

                # for i, item in enumerate(source_batch):
                #     ids, span1, span2, rel = item[1], item[3], item[4], item[5]
                #     item_key = get_key(ids, span1, span2)
                #     item_value = {'pred':predicted[i], 'label':rel}
                #     cv_pred_res[item_key] = item_value

            print('{:=^75}'.format(f' Fold:{fold+1} ({CONF["test_set"]}) '))
            show_time()
            
            prf = compute_f1(gold_all, predicted_all)
            for k, v in prf.items():
                print('{}: {:.6f}'.format(k, v))
                cv_eval_data[k].append(v)
            
    # # 保存模型对所有事件对的预测以及事件对的真实标签
    # file_name = f"{type(model).__name__}_pid{os.getpid()}_pred.pickle"
    # with open(f'log/案例分析/{file_name}', 'wb') as f:
    #     pickle.dump(cv_pred_res, f)

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






    



    
    

    
