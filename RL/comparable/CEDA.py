"""
    实现对比算法ceda
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
    df['ru'] = -1

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
    return df[df['parents'].apply(lambda x : len(x) == 0)].index.tolist()

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

def compute_vm(df, machine_config, path, type=None):
    return compute_vm_by_time(df, machine_config, path)

def compute_vm_by_cost(df, machine_config, task):
    machines = machine_config
    sum = 0
    min = 100000000
    min_sum = 0
    vm = -1
    for machine in machines:
        m_id = machine.id
        est = df.loc[task]['EST']
        sum = max(est, vm_run_time[m_id][-1][1])
        eft = sum + task_run_time[task][m_id]
        if eft < df.loc[task, 'LFT']:
            cost = task_run_time[task][m_id] * machine.price
            if min > cost:
                min = cost
                vm = m_id
                min_sum = sum
            elif abs(cost - min) < 0.001 * min:
                if min_sum > sum:
                    min = cost
                    vm = m_id
                    min_sum = sum
    if vm >= 0:
        vm_run_time[vm].append((min_sum, min_sum+task_run_time[task][vm]))
    else:
        # to do
        # 如果预留池中不满足条件，就要从租赁池中进行租赁
        print('to do')
        exit(1)
    return vm

def compute_vm_by_time(df, machine_config, task):
    machines = machine_config
    sum = 0
    min = 100000000
    min_sum = 0
    vm = -1
    for machine in machines:
        m_id = machine.id
        est = df.loc[task]['EST']
        sum = max(est, vm_run_time[m_id][-1][1])
        eft = sum + task_run_time[task][m_id]
        if eft < df.loc[task, 'LFT']:
            cost = task_run_time[task][m_id]
            if min > cost:
                min = cost
                vm = m_id
                min_sum = sum
            # elif abs(cost - min) < 0.001 * min:
            #     if min_sum > sum:
            #         min = cost
            #         vm = m_id
            #         min_sum = sum
    if vm >= 0:
        vm_run_time[vm].append((min_sum, min_sum+task_run_time[task][vm]))
    else:
        # to do
        # 如果预留池中不满足条件，就要从租赁池中进行租赁
        print('to do')
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

def compute_ru(df, start_task):
    stack = []
    stack += start_task
    while len(stack) > 0:
        task = stack[-1]
        children = df.loc[task, 'children']
        if len(children) == 0:
            df.loc[task, 'ru'] = df.loc[task, 'MET']
            stack.pop()
        else:
            pop_or_not = True
            ru_time = -1
            for child in children:
                if df.loc[child, 'ru'] < 0:
                    stack.append(child)
                    pop_or_not = False
                    break
                else:
                    ru_time = max(ru_time, df.loc[child, 'ru'])
            if not pop_or_not:
                continue
            df.loc[task, 'ru'] = ru_time
            stack.pop()

def assign_vm_to_task(df, sorted_tasks):
    for task in sorted_tasks:
        vm  = compute_vm(df, machine_config, task)
        df.loc[task, 'VmId'] = vm
        compute_est_eft(df, task)
        compute_lft(df, task)

if __name__=='__main__':
    df = read_csv(df, os.getcwd() + '/csv_2/10/CyberShake_10_1.csv')
    compute_runtime(df)
    start_job = initial_EST_EFT(df)
    end_task = initial_LFT(df)
    compute_ru(df, start_job)
    sorted_task_ids_desc = df.sort_values(by='ru', ascending=False).index.tolist()
    assign_vm_to_task(df, sorted_task_ids_desc)
    print()