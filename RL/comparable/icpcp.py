"""
    实现对比算法icpcp
"""
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
machine_config = [MachineConfig(1, 2048, 1, mips=20*math.pow(4, i), price=1*(i+1)) for i in range(machine_num)]

# 元组list形式 [(start_time, end_time), (),()]
vm_run_time = [[(0,0)] for _ in machine_config]
task_run_time = {}

df = None


def read_csv(df, file_name):
    mips = np.mean([machine.mips for machine in machine_config])
    df = pd.read_csv(file_name)
    df['job_id'] = df['job_id'].astype(str)
    df['task_id'] = df['task_id'].astype(str)
    df['parents'] = df['parents'].apply(ast.literal_eval)
    df['EST'] = 0.0
    df['EFT'] = 0.0
    df['MET'] = df['memory'] / mips
    df['LFT'] = 250
    df['VmId'] = -1
    df['children'] = [[] for _ in range(len(df))]

    children_map = defaultdict(list)

    for _, row in df.iterrows():
        for parent in row['parents']:
            children_map[parent].append(row['task_id'])

    # 更新 DataFrame 的 'children' 列
    df['children'] = df['job_id'].map(children_map)
    df.set_index('task_id', inplace=True)
    return df

def compute_runtime(df):
    for idx, row in df.iterrows():
        task_id = idx
        memory = row['memory']
        task_run_time[task_id] = [memory/machine.mips for machine in machine_config]
    return 

def initial_EST_EFT(df):
    # 处理先处理start节点，再处理孩子节点
    start_job = df[df['parents'].apply(lambda x : len(x) == 0)]['job_id'].values.tolist()
    df.loc[df['job_id'].isin(start_job), 'EFT'] = df['MET']
    queue = deque(start_job)
    while len(queue) > 0:
        job_id = queue.popleft()
        job_eft = df[df['job_id']==job_id]['EFT'].values[0]
        # 这里算的孩子节点
        tasks_id = df['parents'].apply(lambda x: job_id in x)
        df['EST'] = np.where(
                        tasks_id,  # 条件：parents 列包含 job_id
                        np.maximum(df['EST'], job_eft),  # 如果条件满足，则取 EST 和 job_eft 的最大值
                        df['EST']  # 否则保持原值不变
                    )
        df.loc[tasks_id, 'EFT'] = df['EST'] + df['MET']
        children = df[tasks_id]['job_id'].values.tolist()
        queue += children
    df['initial_EST'] = df['EST']
    df['initial_EFT'] = df['EFT']
    df['initial_MET'] = df['MET']
    return start_job

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
        children_LFT = df.loc[df['parents'].apply(lambda x : job_id in x)]
        if len(children_LFT) > 0:
            df.loc[job, 'LFT'] = np.min(children_LFT['LFT']-children_LFT['MET'])
        for i in df.loc[job, 'parents'].values.tolist():
            queue += i
    end_task = df.loc[df['job_id'].isin(jobs_list)].index.tolist()
    df['initial_LFT'] = df['LFT']
    return end_task

def assign_parents(df, task_id):
    while True:
        path = []
        children = df.loc[task_id, 'children']
        if df.loc[task_id, 'VmId'] < 0 and len(children) == 0:
            path.append(task_id)
        path = assign_path(df, path, task_id)
        if len(path) == 0:
            break
        # 计算满足关键路径的vm
        vmid = compute_vm(df, machine_config, path)
        for t in path:
            df.loc[t, 'VmId'] = vmid
            compute_est_eft(df, t)
            compute_lft(df, t)
        for t in path:
            assign_parents(df, t)

def assign_path(df, path, task_id):
    parents = df.loc[task_id, 'parents']
    parents = df.loc[df['job_id'].isin(parents)]
    max_eft_task = None
    parents = parents[parents['VmId'] == -1]
    if len(parents) == 0:
        return path
    max_eft_task = parents['EFT'].idxmax()
    if max_eft_task is not None:
        path.append(max_eft_task)
        path = assign_path(df, path, max_eft_task)
    return path

def compute_vm(df, machine_config, path, type=None):
    return compute_vm_by_time(df, machine_config, path)

def compute_vm_by_cost(df, machine_config, path):
    machines = machine_config
    sum = 0
    sum_init = 0
    min = 100000000
    deadline = 0
    vm = -1
    for machine in machines:
        m_id = machine.id
        est = df.loc[path[-1]]['EST']
        sum = max(est, vm_run_time[m_id][-1][1])
        sum_init = sum
        match = True
        for t in reversed(path):
            sum += task_run_time[t][m_id]
            if (sum > df.loc[t]['LFT']):
                match = False
                break
        if not match:
            continue
        cost = (sum - sum_init)*machine_config[m_id].price
        if min > cost:
            deadline = sum
            min = cost
            vm = m_id
    if vm >= 0:
        vm_run_time[vm].append((sum_init, deadline))
    else:
        # to do
        # 如果预留池中不满住条件，就要从租赁池中进行租赁
        print('to do')
        exit(1)
    return vm

def compute_vm_by_time(df, machine_config, path):
    machines = machine_config
    task = path[-1]
    all_candidates = []
    sum = 0
    sum_init = 0
    min = 100000000
    deadline = 0
    vm = -1
    for machine in machines:
        m_id = machine.id
        est = df.loc[task]['EST']
        sum = max(est, vm_run_time[m_id][-1][1])
        sum_init = sum
        match = True
        for t in reversed(path):
            sum += task_run_time[t][m_id]
            if (sum > df.loc[t]['LFT']):
                match = False
                break
        if not match:
            continue
        cost = sum
        if min > cost:
            deadline = sum
            min = cost
            vm = m_id
    if vm >= 0:
        vm_run_time[vm].append((sum_init, deadline))
    else:
        # to do
        # 如果预留池中不满住条件，就要从租赁池中进行租赁
        exit(1)
    return vm

def compute_est_eft(df, task):
    queue = deque([task])
    while len(queue) > 0:
        t = queue.popleft()
        # 找到父节点中最大的eft，更新
        parents = df.loc[t, 'parents']
        if len(parents) > 0:
            df.loc[t, 'EST'] = df.loc[df['job_id'].isin(parents), 'EFT'].max()
        if df.loc[t, 'VmId'] > -1:
            df.loc[t, 'MET'] = df.loc[t, 'memory']/machine_config[df.loc[t, 'VmId']].mips
        df.loc[t, 'EFT'] = df.loc[t, 'EST'] + df.loc[t, 'MET']
        # 找到孩子节点，孩子节点入队
        children = df.loc[t, 'children']
        queue += children

def compute_lft(df, task):
    queue = deque([task])
    while len(queue) > 0:
        t = queue.popleft()
        children = df.loc[t, 'children']
        if df.loc[t, 'VmId'] > -1:
            df.loc[t, 'MET'] = df.loc[t, 'memory']/machine_config[df.loc[t, 'VmId']].mips
        if len(children) > 0:
            df.loc[t, 'LFT'] = df.loc[children, 'LFT'].max() - df.loc[t, 'MET']
        parents = df.loc[t, 'parents']
        parents = df.loc[df['job_id'].isin(parents)].index.tolist()
        queue += parents


if __name__=='__main__':
    # task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_type = ['CyberShake']
    job_num = 10
    file_num = 10
    task_num = [10]
    for ty in task_type:
        for t_num in task_num:
            for i in range(file_num):
                job_file = os.getcwd() + '/csv_2/{}/{}_{}_{}.csv'.format(t_num, ty, job_num, i)
                df = read_csv(df, job_file)
                compute_runtime(df)
                start_jobs = initial_EST_EFT(df)
                end_tasks = initial_LFT(df)
                for end in end_tasks:
                    assign_parents(df, end)
                print()