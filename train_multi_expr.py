import numpy as np
from itertools import product

import train_ESC
import train_CTB
from config import CONF

if __name__=='__main__':

    data_seeds = CONF['data_seeds']
    model_seeds = CONF['model_seeds']
    data_set = CONF['data_set']
    data_v = CONF['data_version']

    static_dict = {}
    iter = CONF['iter'] if 'iter' in CONF and CONF['iter']!=[] else product(model_seeds, data_seeds)
    for ms, ds in iter:
        CONF['model_seed'] = ms
        CONF['data_seed'] = ds
        key, value = f'ms_{ms} ds_{ds}', None
        if data_set == 'ESC':
            CONF['data_path'] = f'./data/data_v{data_v}_ESC_{ds}.pickle'
            value = train_ESC.train_and_eval(CONF)
            static_dict[key] = value

            # 保存模型在不同attn_head时的模型表现
            # for attn_head in range(2, 18, 2):
            #     CONF['attn_head'] = attn_head
            #     value = train_v2.train_and_eval(CONF)
            #     static_dict[f'ms_{ms} ds_{ds} head_{attn_head}'] = value
        else:
            CONF['data_path'] = f'./data/data_v4.2_CTB_{ds}.pickle' 
            value = train_CTB.train_and_eval(CONF)
            static_dict[key] = value

    # Calculate and print the average of P, R, F
    print('{:=^75}'.format(f' All Expr Info '))
    print('metrics\t\t: precision, recall, f1-score')
    data_summary = []
    for k, prf in static_dict.items():
        data_summary.extend(prf.values())
        print(f'{k}\t: '+', '.join(['{:.6f}'.format(v) for v in prf.values()]))
    data_summary = np.array(data_summary).reshape(-1, 3)
    avg = np.mean(data_summary, 0)
    print('Average\t\t: '+', '.join(['{:.6f}'.format(v) for v in avg]))


    