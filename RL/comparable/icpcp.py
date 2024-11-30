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

machine_config = {'stay':[], 'rent':[]}
machine_type = {'F2':[2, 4, 0.192, 0.1539], 'F4':[4, 8, 0.383, 0.3099], 'F8':[8, 16, 0.766, 0.6158]}
machine_num = {'F2': [4, 4], 'F4':[4, 4], 'F8':[4, 4]}
total_machine = []

def create_vm(machine_type, machine_num, total_machine):
    for key, value in machine_num.items():
        m_type = machine_type[key]
        stay_machine = [MachineConfig(1, m_type[1]*1024, 1, mips=10*m_type[0], price=m_type[3]) for _ in range(value[0])]
        machine_config['stay'] += stay_machine
        total_machine += stay_machine
    
    for key, value in machine_num.items():
        m_type = machine_type[key]
        rent_machine = [MachineConfig(1, m_type[1]*1024, 1, mips=10*m_type[0], price=m_type[3]) for _ in range(value[1])]
        machine_config['rent'] += rent_machine
        total_machine += rent_machine

def read_csv(file_name, deadline=1000.0):
    mips = np.max([machine.mips for machine in machine_config['stay']])
    df = pd.read_csv(file_name)
    df['job_id'] = df['job_id'].astype(str)
    df['task_id'] = df['task_id'].astype(str)
    df['parents'] = df['parents'].apply(ast.literal_eval)
    df['EST'] = 0.0
    df['EFT'] = 0.0
    df['MET'] = df['memory'] / mips
    df['LFT'] = deadline
    df['AST'] = deadline
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

class ICPCP(object):

    def __init__(self, df, machine_config, total_machine) -> None:
        self.machine_config = machine_config
        self.total_machine = total_machine
        self.vm_run_time = [[(0,0)] for _ in total_machine]
        self.task_run_time = {}
        self.df = df
        self.compute_runtime()
        self.deadline = 0

    def compute_runtime(self):
        for idx, row in self.df.iterrows():
            task_id = idx
            memory = row['memory']
            self.task_run_time[task_id] = [memory/machine.mips for machine in total_machine]

    def initial_EST_EFT(self):
        # 处理先处理start节点，再处理孩子节点
        start_job = self.df[self.df['parents'].apply(lambda x : len(x) == 0)]['job_id'].values.tolist()
        self.df.loc[self.df['job_id'].isin(start_job), 'EFT'] = self.df['MET']
        queue = deque(start_job)
        while len(queue) > 0:
            job_id = queue.popleft()
            job_eft = self.df[self.df['job_id']==job_id]['EFT'].values[0]
            # 这里算的孩子节点
            tasks_id = self.df['parents'].apply(lambda x: job_id in x)
            self.df['EST'] = np.where(
                            tasks_id,  # 条件：parents 列包含 job_id
                            np.maximum(self.df['EST'], job_eft),  # 如果条件满足，则取 EST 和 job_eft 的最大值
                            self.df['EST']  # 否则保持原值不变
                        )
            self.df.loc[tasks_id, 'EFT'] = self.df['EST'] + self.df['MET']
            children = self.df[tasks_id]['job_id'].values.tolist()
            queue += children
        self.df['initial_EST'] = self.df['EST']
        self.df['initial_EFT'] = self.df['EFT']
        self.df['initial_MET'] = self.df['MET']
        return start_job

    def initial_LFT(self):
        jobs_id = self.df['job_id'].values.tolist()
        parents = []
        for i in self.df['parents'].values:
            parents += i
        jobs_list = list(set(jobs_id)-set(parents))
        queue = deque(jobs_list)
        while len(queue) > 0:
            job_id = queue.popleft()
            job = self.df['job_id'] == job_id
            children_LFT = self.df.loc[self.df['parents'].apply(lambda x : job_id in x)]
            if len(children_LFT) > 0:
                self.df.loc[job, 'LFT'] = np.min(children_LFT['LFT']-children_LFT['MET'])
            for i in self.df.loc[job, 'parents'].values.tolist():
                queue += i
        end_task = self.df.loc[self.df['job_id'].isin(jobs_list)].index.tolist()
        self.df['initial_LFT'] = self.df['LFT']
        return end_task

    def assign_parents(self, task_id, type='time'):
        while True:
            path = []
            children = self.df.loc[task_id, 'children']
            if self.df.loc[task_id, 'VmId'] < 0 and len(children) == 0:
                path.append(task_id)
            path = self.assign_path(path, task_id)
            if len(path) == 0:
                break
            # 计算满足关键路径的vm
            vmid = self.compute_vm(machine_config['stay'], path, type=type)
            if vmid < 0:
                vmid = self.compute_vm(machine_config['rent'], path, type=type)

            for t in path:
                df.loc[t, 'VmId'] = vmid
                self.compute_est_eft(t)
                self.compute_lft(t)
            for t in path:
                self.assign_parents(t, type=type)

    def assign_path(self, path, task_id):
        parents = self.df.loc[task_id, 'parents']
        parents = self.df.loc[df['job_id'].isin(parents)]
        max_eft_task = None
        parents = parents[parents['VmId'] == -1]
        if len(parents) == 0:
            return path
        max_eft_task = parents['EFT'].idxmax()
        if max_eft_task is not None:
            path.append(max_eft_task)
            path = self.assign_path(path, max_eft_task)
        return path

    def compute_vm(self, machines, path, type=None):
        if type == 'time':
            return self.compute_vm_by_time(machines, path)
        else:
            return self.compute_vm_by_cost(machines, path)

    def compute_vm_by_cost(self, machine_config, path):
        machines = machine_config
        sum = 0
        sum_init = 0
        min = 100000000
        deadline = 0
        vm = -1
        for machine in machines:
            m_id = machine.id
            est = self.df.loc[path[-1]]['EST']
            sum = max(est, self.vm_run_time[m_id][-1][1])
            est = sum
            match = True
            for t in reversed(path):
                sum += self.task_run_time[t][m_id]
                if (sum > self.df.loc[t]['LFT']):
                    match = False
                    break
            if not match:
                continue
            cost = (sum - est)*total_machine[m_id].price
            if min > cost:
                deadline = sum
                min = cost
                vm = m_id
                sum_init = est
        if vm >= 0:
            self.deadline = max(self.deadline, deadline)
            self.vm_run_time[vm].append((sum_init, deadline))
        else:
            # to do
            # 如果预留池中不满住条件，就要从租赁池中进行租赁
            print('to rent')
            return -1
        return vm

    def compute_vm_by_time(self, machine_config, path):
        machines = machine_config
        task = path[-1]
        sum = 0
        sum_init = 0
        min = 100000000
        deadline = 0
        vm = -1
        for machine in machines:
            m_id = machine.id
            est = self.df.loc[task]['EST']
            sum = max(est, self.vm_run_time[m_id][-1][1])
            est = sum
            match = True
            for t in reversed(path):
                sum += self.task_run_time[t][m_id]
                if (sum > self.df.loc[t]['LFT']):
                    match = False
                    break
            if not match:
                continue
            cost = sum
            if min > cost:
                deadline = sum
                min = cost
                vm = m_id
                sum_init = est
        if vm >= 0:
            self.deadline = max(self.deadline, deadline)
            self.vm_run_time[vm].append((sum_init, deadline))
        else:
            # to do
            # 如果预留池中不满住条件，就要从租赁池中进行租赁
            print('to rent')
            return -1
        return vm

    def compute_est_eft(self, task):
        queue = deque([task])
        while len(queue) > 0:
            t = queue.popleft()
            # 找到父节点中最大的eft，更新
            parents = self.df.loc[t, 'parents']
            if len(parents) > 0:
                self.df.loc[t, 'EST'] = self.df.loc[self.df['job_id'].isin(parents), 'EFT'].max()
            if self.df.loc[t, 'VmId'] > -1:
                self.df.loc[t, 'MET'] = self.df.loc[t, 'memory']/total_machine[self.df.loc[t, 'VmId']].mips
            self.df.loc[t, 'EFT'] = self.df.loc[t, 'EST'] + self.df.loc[t, 'MET']
            # 找到孩子节点，孩子节点入队
            children = self.df.loc[t, 'children']
            queue += children

    def compute_lft(self, task):
        df = self.df
        queue = deque([task])
        while len(queue) > 0:
            t = queue.popleft()
            children = df.loc[t, 'children']
            if df.loc[t, 'VmId'] > -1:
                df.loc[t, 'MET'] = df.loc[t, 'memory']/total_machine[df.loc[t, 'VmId']].mips
            if len(children) > 0:
                df.loc[t, 'LFT'] = df.loc[children, 'LFT'].max() - df.loc[t, 'MET']
            parents = df.loc[t, 'parents']
            parents = df.loc[df['job_id'].isin(parents)].index.tolist()
            queue += parents
        self.df = df


if __name__=='__main__':
    create_vm(machine_type, machine_num, total_machine)
    # task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_type = ['CyberShake']
    job_num = 10
    file_num = 1000
    task_num = [10]
    for ty in task_type:
        for t_num in task_num:
            for i in range(file_num):
                job_file = os.getcwd() + '/csv_2/{}/{}_{}_{}.csv'.format(t_num, ty, job_num, i)
                df = read_csv(job_file)
                icpcp = ICPCP(df, machine_config, total_machine)
                start_jobs = icpcp.initial_EST_EFT()
                end_tasks = icpcp.initial_LFT()
                for end in end_tasks:
                    icpcp.assign_parents(end)
                icpcp.df['LFT'] += icpcp.deadline - 1000
                icpcp.df.to_csv(os.getcwd() + '/csv_pretrain/{}/{}_{}_{}.csv'.format(t_num, ty, job_num, i))
                print()