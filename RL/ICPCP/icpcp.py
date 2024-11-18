
import os
import time
import numpy as np
import pandas as pd
import torch
import sys
from matplotlib import pyplot as plt
import math
import ast
from collections import deque
from collections import defaultdict

sys.path.append('./')

from CloudSimPy.core.machine import MachineConfig

machine_num=2
machine_config = [MachineConfig(1, 1024, 1, mips=2*math.pow(10, i), price=1*(i+1)) for i in range(machine_num)]

def read_csv(file_name):
    mips = np.min([machine.mips for machine in machine_config])
    df = pd.read_csv(file_name)
    df['job_id'] = df['job_id'].astype(str)
    df['task_id'] = df['task_id'].astype(str)
    df['parents'] = df['parents'].apply(ast.literal_eval)
    df['EST'] = 0.0
    df['EFT'] = 0.0
    df['MET'] = df['memory'] / mips
    df['LFT'] = 0.0
    df['vmId'] = -1
    df['children'] = [[] for _ in range(len(df))]

    children_map = defaultdict(list)

    for _, row in df.iterrows():
        for parent in row['parents']:
            children_map[parent].append(row['job_id'])

    # 更新 DataFrame 的 'children' 列
    df['children'] = df['job_id'].map(children_map)
    return df

def compute_runtime(df):
    
    return 

def initial_EST_EFT(df):
    # 处理先处理start节点，再处理孩子节点
    start_job = df[df['parents'].apply(lambda x : len(x) == 0)]['job_id'].values.tolist()
    df.loc[df['job_id'].isin(start_job), 'EFT'] = df['MET']
    queue = deque(start_job)
    while len(queue) > 0:
        job_id = queue.popleft()
        job_eft = df[df['job_id']==job_id]['EFT'].values[0]
        tasks_id = df['parents'].apply(lambda x: job_id in x)
        df['EST'] = np.where(
                        tasks_id,  # 条件：parents 列包含 job_id
                        np.maximum(df['EST'], job_eft),  # 如果条件满足，则取 EST 和 job_eft 的最大值
                        df['EST']  # 否则保持原值不变
                    )
        df.loc[tasks_id, 'EFT'] = df['EST'] + df['MET']
        children = df[tasks_id]['job_id'].values.tolist()
        queue += children
    return df, start_job

def initial_LFT(df):
    jobs_id = df['job_id'].values.tolist()
    parents = []
    for i in df['parents'].values:
        parents += i
    jobs_list = list(set(jobs_id)-set(parents))
    queue = deque(jobs_list)
    while len(queue) > 0:
        job_id = queue.popleft()
        job = df['job_id'] == job_id
        children_LFT = df.loc[df['parents'].apply(lambda x : job_id in x), 'LFT'].values
        if len(children_LFT) > 0:
            df.loc[job, 'LFT'] = np.min(children_LFT)-df.loc[job, 'MET']
        for i in df.loc[job, 'parents'].values.tolist():
            queue += i
    return df, jobs_list

def assign_parents(df, task_id, path):
    while True:
        path = []
        task = df['task']==task_id
        children = df['children']
        if df.loc[task, 'VmId'] >= 0 and len(children) == 0:
            path.append(task_id)
        assign_path(df, path, task_id)

def assign_path(df, path, task_id):
    parents = df.loc[df['task_id']==task_id, 'parents'].values.tolist()
    parents = df.loc[df['job_id'].isin(parents)]
    max_eft_task = parents.loc[parents['EFT'].idxmax(), 'task_id']
    if max_eft_task is not 
    path.add(max_eft_task)


if __name__=='__main__':
    df = read_csv(os.getcwd() + '/csv_2/10/CyberShake_10_0.csv')
    df, start_jobs = initial_EST_EFT(df)
    df, end_jobs = initial_LFT(df)