# 知识驱动的事件因果关系识别
+ KeAGN模型全称为Knowledge-enriched and Attention Guided Network，即富知识化与注意力引导网络
+ KeAGN模型由本人在毕业论文中提出，其主要应用于NLP领域中的句级因果关系识别
+ KeAGN模型改进自LSIN模型，LSIN模型详情可见[Knowledge-Enriched Event Causality Identification via Latent Structure Induction Networks](https://aclanthology.org/2021.acl-long.376/)

## Step 0: 准备数据  
1. 下载数据集 (建议略过该步, 直接进行[2]和Step 3)  
    [EventStoryLine](https://github.com/tommasoc80/EventStoryLine)  
    [Causal-TimeBank](https://github.com/paramitamirza/Causal-TimeBank)
2. 下载论文作者使用的数据 (包含数据集、零跳概念匹配数据、ConceptNet数据、COMET生成数据)  
    [全部数据(Google Drive)](https://drive.google.com/drive/folders/1juvVPa7wqYqBYzj-wvpwbBKV3Jkufk11?usp=sharing)  
3. 解压覆盖  
    步骤[1]-->解压文件夹, 将数据集复制到`data`文件夹下  
    步骤[2]-->解压文件夹至项目文件夹下，覆盖原来的`data`文件夹

PS: 完成[2]之后可以直接进行**Step 3**

## Step 1: 从XML文件中提取数据
```
python read_document_ESC.py
python read_document_CTB.py
```

## Step 2: 知识引入
```
python preprocess.py
```
**PS:** 文件中的`seed`用于修改随机数种子, `ds`用于修改要处理的数据集

## Step 3: 模型训练及评估
训练之前需要在`config.py`文件中配置相关信息  
配置完成后运行`train.sh`，示例及备注如下：
```
./train.sh KeAGN_epoch15 me
``` 
1. `KeAGN`为输出文件名, 可自行修改, shell文件会自动补充后缀`.log`  
2. `me`为实验方式, 可修改为`ESC`或`CTB`  
    `me`表示多次实验求平均值, 实验数据集  
    `ESC`表示在`EventStoryLine`数据集上进行单次实验  
    `CTB`表示在`Causal-TimeBank`数据集上进行单次实验
