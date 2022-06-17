CONF = {
    # 1. 主要参数 
    'model_name': 'MixedModel(bert=True, dk=True, attn=True, rk=True, rk_aggcn=True, hn_list=[200])', # 可以选择MixedModel与BaseBerModel, 参数参考`models`中的`model.py`和`LSIN.py`
    'cuda_index': 1, # GPU设备索引号
    'note': "'获取分析案例'", # 实验备注
    'data_set': 'ESC', # 实验数据集
    'neg_sam_rate': 0.45, # ESC为0.45, CTB为0.3
    'data_version': '4.2', # 无需修改
    'test_set': 'test', # 可以选择验证集'dev'或者测试集'test',

    # 2. 多次实验求平均值时需要指定的参数:
    'data_seeds': [0,1234,2022], # 数据预处理的随机种子
    'model_seeds': [2022,1234,0], # 模型训练的随机种子
    'iter': [], # 自定义种子组合, 示例：[(2022,2022),(0,0)]

    # 3. 单次实验需要指定的参数
    'data_path': 'data/data_v4.2_ESC_0.pickle',
    'model_seed': 2022, # 模型训练阶段的随机种子
    'data_seed': 0, # 知识引入即数据预处理阶段的随机种子

    # 4. 训练相关的参数
    'train_epoch': 15, # 在训练集上的迭代轮数
    'batch_size': 20, # 无需修改
    'learning_rate': 2e-5, # 无需修改
    'bert_path': 'bert-base-uncased' # 若本地有HuggingFace发布的权重文件，可自行配置为/xx..x/bert-base-uncased
}


def show_config():
    """打印配置信息"""
    
    print('{:=^75}'.format(' Config Info '))
    for key in CONF:
        print(key+":", CONF[key])
    print('='*75)


if __name__ == '__main__':
    # show_config()
    import json
    print(json.dumps(CONF, indent=4))