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

machine_config = {'stay':[], 'rent':[]}
machine_type = {'F2':[2, 4, 0.192, 0.1539], 'F4':[4, 8, 0.383, 0.3099], 'F8':[8, 16, 0.766, 0.6158]}
machine_num = {'F2': [4, 4], 'F4':[4, 4], 'F8':[4, 4]}
total_machine = []

df = None

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
    mips = np.mean([machine.mips for machine in machine_config['stay']])
    df = pd.read_csv(file_name)
    df['job_id'] = df['job_id'].astype(str)
    df['task_id'] = df['task_id'].astype(str)
    df['parents'] = df['parents'].apply(ast.literal_eval)
    df['EST'] = 0
    df['EFT'] = 0
    df['MET'] = np.ceil(df['memory'] / mips).astype(int)
    df['LFT'] = deadline
    df['AST'] = deadline
    df['VmId'] = -1
    df['ru'] = -1.0
    df['children'] = [[] for _ in range(len(df))]

    children_map = defaultdict(list)

    for _, row in df.iterrows():
        for parent in row['parents']:
            children_map[parent].append(row['task_id'])

    # 更新 DataFrame 的 'children' 列
    df['children'] = df['job_id'].map(children_map)
    df.set_index('task_id', inplace=True)
    return df

class CEDA(object):

    def __init__(self, df, machine_config, total_machine) -> None:
        self.machine_config = machine_config
        self.total_machine = total_machine
        self.vm_run_time = [[(0,0)] for _ in total_machine]
        self.task_run_time = {}
        self.df = df
        self.compute_runtime()
        self.deadline = 0


    def compute_runtime(self):
        for idx, row in df.iterrows():
            task_id = idx
            memory = row['memory']
            self.task_run_time[task_id] = [np.ceil(memory/machine.mips).astype(int) for machine in total_machine]
        return 

    def initial_EST_EFT(self):
        df = self.df
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
        self.df = df
        return df[df['parents'].apply(lambda x : len(x) == 0)].index.tolist()

    def initial_LFT(self):
        df = self.df
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
        self.df = df
        return end_task

    def compute_vm(self, machine_config, path, type=None):
        return self.compute_vm_by_time(machine_config, path)

    def compute_vm_by_cost(self, machine_config, task):
        df = self.df
        machines = machine_config
        sum = 0
        min = 100000000
        min_sum = 0
        vm = -1
        for machine in machines:
            m_id = machine.id
            est = df.loc[task]['EST']
            sum = max(est, self.vm_run_time[m_id][-1][1])
            eft = sum + self.task_run_time[task][m_id]
            if eft < df.loc[task, 'LFT']:
                cost = self.task_run_time[task][m_id] * machine.price
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
            self.deadline = max(self.deadline, min_sum+self.task_run_time[task][vm])
            self.vm_run_time[vm].append((min_sum, min_sum+self.task_run_time[task][vm]))
        else:
            # to do
            # 如果预留池中不满足条件，就要从租赁池中进行租赁
            print('to do')
            exit(1)
        return vm

    def compute_vm_by_time(self, machine_config, task):
        machines = machine_config
        sum = 0
        min = 100000000
        min_sum = 0
        vm = -1
        df = self.df
        for machine in machines:
            m_id = machine.id
            est = df.loc[task]['EST']
            sum = max(est, self.vm_run_time[m_id][-1][1])
            eft = sum + self.task_run_time[task][m_id]
            if eft < df.loc[task, 'LFT']:
                cost = eft
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
            self.deadline = max(self.deadline, min_sum + self.task_run_time[task][vm])
            self.vm_run_time[vm].append((min_sum, min_sum + self.task_run_time[task][vm]))
            self.df.loc[task, 'EST'] = min_sum
            # print('task: ', task, ' start in vm ', vm, ' time', min_sum)
        else:
            # to do
            # 如果预留池中不满足条件，就要从租赁池中进行租赁
            print('to do')
            exit(1)
        return vm

    def compute_est_eft(self, task):
        df = self.df
        queue = deque([task])
        while len(queue) > 0:
            t = queue.popleft()
            # 找到父节点中最大的eft，更新
            parents = df.loc[t, 'parents']
            if len(parents) > 0:
                df.loc[t, 'EST'] = max(df.loc[df['job_id'].isin(parents), 'EFT'].max(), df.loc[t, 'EST'])
            if df.loc[t, 'VmId'] > -1:
                df.loc[t, 'MET'] = np.ceil(df.loc[t, 'memory']/total_machine[df.loc[t, 'VmId']].mips).astype(int)
            df.loc[t, 'EFT'] = df.loc[t, 'EST'] + df.loc[t, 'MET']
            # 找到孩子节点，孩子节点入队
            children = df.loc[t, 'children']
            queue += children
        self.df = df

    def compute_lft(self, task):
        df = self.df
        queue = deque([task])
        while len(queue) > 0:
            t = queue.popleft()
            children = df.loc[t, 'children']
            if df.loc[t, 'VmId'] > -1:
                df.loc[t, 'MET'] = np.ceil(df.loc[t, 'memory']/total_machine[df.loc[t, 'VmId']].mips).astype(int)
            if len(children) > 0:
                df.loc[t, 'LFT'] = df.loc[children, 'LFT'].max() - df.loc[t, 'MET']
            parents = df.loc[t, 'parents']
            parents = df.loc[df['job_id'].isin(parents)].index.tolist()
            queue += parents
        self.df = df

    def compute_ru(self, start_task):
        df = self.df
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
                df.loc[task, 'ru'] = ru_time + df.loc[task, 'MET']
                stack.pop()
        
        self.df = df

    def assign_vm_to_task(self, sorted_tasks):
        for task in sorted_tasks:
            vm  = self.compute_vm(machine_config['stay'], task)
            if vm < 0:
                vm = self.compute_vm(machine_config['rent'], task)
            # print(task, ' run on ', vm)
            df.loc[task, 'VmId'] = vm
            self.compute_est_eft(task)
            self.compute_lft(task)

if __name__=='__main__':
    create_vm(machine_type, machine_num, total_machine)
    # task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_type = ['Genome']
    job_num = 10
    file_num = 5000
    task_num = [10]

    for ty in task_type:
        for t_num in task_num:
            for i in range(463, file_num):
                job_file = os.getcwd() + '/csv_2/{}/{}/{}_{}_{}.csv'.format(t_num, ty, ty, job_num, i)
                print('job num ', job_num, i)
                df = read_csv(job_file)
                ceda = CEDA(df, machine_config, total_machine)
                start_jobs = ceda.initial_EST_EFT()
                end_tasks = ceda.initial_LFT()
                ceda.compute_ru(start_jobs)

                sorted_task_ids_desc = ceda.df.sort_values(by='ru', ascending=False).index.tolist()
                ceda.assign_vm_to_task(sorted_task_ids_desc)
                ceda.df['LFT'] += ceda.deadline - 1000
                # 按 'EST' 从小到大排序，根据ru从大到小进行排序，根据MET从小到大进行排序，并生成唯一的 rankScore
                sorted_df = ceda.df.sort_values(by=['EST', 'ru'], ascending=[True, False])
                # 为排序后的 DataFrame 添加 rankScore
                sorted_df['rankScore'] = range(1, len(ceda.df) + 1)
                # 恢复原来的索引
                ceda.df['rankScore'] = sorted_df['rankScore']
                ceda.df.to_csv(os.getcwd() + '/csv_pretrain_ceda/{}/{}/{}_{}_{}.csv'.format(t_num, ty, ty, job_num, i))
                print()