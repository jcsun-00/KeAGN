#!/bin/bash

# 启动conda环境
conda_path="/data/lujun/miniconda3/"
env_name="LSIN-jcsun"
source "${conda_path}/bin/activate" "$env_name"

# 确保存在$1
test -z $1 && echo "No file name." && exit 0

# 生成log文件名及保存地址
log_file="./log/$1.log" 
touch "${log_file}"
echo "log_file: ${log_file}"

# 运行train.py (以nohup方式运行)
if [ "$2" == "ctb" ]; then
    nohup python -u ./train_CTB.py > ${log_file} & 
elif [ "$2" == "esc" ]; then
    nohup python -u ./train_ESC.py > ${log_file} & 
elif [ "$2" == "me" ]; then
    nohup python -u ./train_multi_expr.py > ${log_file} & 
else
    echo "Specify how to run: ctb, esc or me ?" && exit 0
fi

# 查看日志
tail -f ${log_file}