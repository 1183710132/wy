from operator import attrgetter
import pandas as pd
import numpy as np
import ast


from CloudSimPy.core.job import JobConfig, TaskConfig, TaskInstanceConfig
from CloudSimPy.playground.DAG.utils.feature_synthesize import father_task_indices


'''
    区别于第一版的调度方案，在这v2版的数据映射上，将task_instance映射为task，task映射为job，job映射为整体的调度方案
    目的就是为了可以通过直接调度task，每个task都有各自的memory和cpu需求，方便进行代价计算
'''

class CSVReader(object):
    def __init__(self, filename):
        self.filename = filename
        df = pd.read_csv(self.filename)

        df.job_id = df.job_id.astype(dtype=str)
        df.instances_num = df.instances_num.astype(dtype=int)

        task_instance_map = {}
        job_task_map = {}
        job_submit_time_map = {}
        for i in range(len(df)):
            series = df.iloc[i]
            # 映射
            task_id = series.job_id
            task_instance_id = series.task_id
            parent_indices = ast.literal_eval(series.parents)
            cpu = series.cpu
            memory = series.memory
            disk = series.disk
            duration = series.duration
            submit_time = series.submit_time
            # 修改task instance config文件
            task_instance_configs = task_instance_map.setdefault(task_id, [])
            task_instance_configs.append(TaskInstanceConfig(task_instance_id, cpu, memory, parents_indices=parent_indices))
            job_submit_time_map[task_id] = submit_time

        task_configs = []
        for task_id, task_instance_config in task_instance_map.items():
            task_configs.append(TaskConfig(task_id, task_instance_config))
        
        job_configs = [JobConfig('job_1', 0, task_configs)]
        job_configs.sort(key=attrgetter('submit_time'))

        self.job_configs = job_configs

    def generate(self, offset, number):
        number = number if offset + number < len(self.job_configs) else len(self.job_configs) - offset
        ret = self.job_configs[offset: offset + number]
        the_first_job_config = ret[0]
        submit_time_base = the_first_job_config.submit_time

        # tasks_number = 0
        # task_instances_numbers = []
        # task_instances_durations = []
        # task_instances_cpu = []
        # task_instances_memory = []
        # for job_config in ret:
        #     job_config.submit_time -= submit_time_base
        #     tasks_number += len(job_config.task_configs)
        #     for task_config in job_config.task_configs:
        #         task_instances_numbers.append(task_config.instances_number)
        #         task_instances_durations.extend([task_config.duration] * int(task_config.instances_number))
        #         task_instances_cpu.extend([task_config.cpu] * int(task_config.instances_number))
        #         task_instances_memory.extend([task_config.memory] * int(task_config.instances_number))

        # print('Jobs number: ', len(ret))
        # print('Tasks number:', tasks_number)

        # print('Task instances number mean: ', np.mean(task_instances_numbers))
        # print('Task instances number std', np.std(task_instances_numbers))

        # print('Task instances cpu mean: ', np.mean(task_instances_cpu))
        # print('Task instances cpu std: ', np.std(task_instances_cpu))

        # print('Task instances memory mean: ', np.mean(task_instances_memory))
        # print('Task instances memory std: ', np.std(task_instances_memory))

        # print('Task instances duration mean: ', np.mean(task_instances_durations))
        # print('Task instances duration std: ', np.std(task_instances_durations))

        return ret
