from operator import attrgetter
import pandas as pd
import numpy as np
import sys
import os
import ast
from collections import deque
sys.path.append('CloudSimPy')
from core.job import JobConfig, TaskConfig
from playground.DAG.utils.feature_synthesize import father_task_indices

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
    EST_min = 0
    EST_max = np.max(df['EST'].values)
    EFT_min = np.min(df['EFT'].values)
    EFT_max = np.min(df['EFT'].values)
    # df['EST'] = (df['EST']-EST_min)/(EST_max-EST_min)
    # df['EFT'] = (df['EFT']-EFT_min)/(EFT_max-EFT_min)
    return df

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
    LFT_min = np.min(df['LFT'].values)
    LFT_max = np.max(df['LFT'].values)
    # df['LFT'] = (df['LFT']-LFT_min)/(LFT_max-LFT_min)
    return df

class CSVReader(object):
    def __init__(self, filename, machine_config, deadline=0):
        mips = np.min([machine.mips for machine in machine_config])

        self.filename = filename
        df = pd.read_csv(self.filename)
        df['job_id'] = df['job_id'].astype(str)
        df['task_id'] = df['task_id'].astype(str)
        df.instances_num = df.instances_num.astype(dtype=int)
        df['parents'] = df['parents'].apply(ast.literal_eval)
        df['EST'] = 0.0
        df['EFT'] = 0.0
        df['MET'] = df['memory'] / mips
        df['LFT'] = deadline
        df['vmId'] = -1.0
        df = initial_EST_EFT(df)
        df = initial_LFT(df)
        job_task_map = {}
        job_submit_time_map = {}
        for i in range(len(df)):
            series = df.iloc[i]
            job_id = series.job_id
            task_id = series.task_id
            submit_time = series.submit_time
            task_configs = job_task_map.setdefault(job_id, [])
            task_configs.append(TaskConfig(series))
            job_submit_time_map[job_id] = submit_time

        job_configs = []
        for job_id, task_configs in job_task_map.items():
            job_configs.append(JobConfig(job_id, job_submit_time_map[job_id], task_configs))
        job_configs.sort(key=attrgetter('submit_time'))

        self.job_configs = job_configs

    def generate(self, offset, number):
        number = number if offset + number < len(self.job_configs) else len(self.job_configs) - offset
        ret = self.job_configs[offset: offset + number]
        the_first_job_config = ret[0]
        submit_time_base = the_first_job_config.submit_time

        tasks_number = 0
        task_instances_numbers = []
        task_instances_durations = []
        task_instances_cpu = []
        task_instances_memory = []
        for job_config in ret:
            job_config.submit_time -= submit_time_base
            tasks_number += len(job_config.task_configs)
            for task_config in job_config.task_configs:
                task_instances_numbers.append(task_config.instances_number)
                task_instances_durations.extend([task_config.duration] * int(task_config.instances_number))
                task_instances_cpu.extend([task_config.cpu] * int(task_config.instances_number))
                task_instances_memory.extend([task_config.memory] * int(task_config.instances_number))

        print('Jobs number: ', len(ret))
        print('Tasks number:', tasks_number)

        cpu_mean = np.mean(task_instances_cpu)
        cpu_std = np.std(task_instances_cpu)
        memory_mean = np.mean(task_instances_memory)
        memory_std = np.std(task_instances_memory)
        
        # print('Task instances number mean: ', np.mean(task_instances_numbers))
        # print('Task instances number std', np.std(task_instances_numbers))

        # print('Task instances cpu mean: ', cpu_mean)
        # print('Task instances cpu std: ', cpu_std)

        # print('Task instances memory mean: ', memory_mean)
        # print('Task instances memory std: ', memory_std)

        # print('Task instances duration mean: ', np.mean(task_instances_durations))
        # print('Task instances duration std: ', np.std(task_instances_durations))

        return ret, [np.max(task_instances_cpu), np.min(task_instances_cpu), np.max(task_instances_memory), np.min(task_instances_memory)]
