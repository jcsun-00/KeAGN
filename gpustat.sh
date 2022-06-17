#!/bin/bash
#===================================================
# 用于便捷查询GPU使用情况
#===================================================

# 启动conda环境
source activate LSIN-jcsun

# 执行gpustat命令
watch -n 2 --color gpustat --c