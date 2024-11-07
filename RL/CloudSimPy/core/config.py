class TaskInstanceConfig(object):
    def __init__(self, instance_id, cpu, memory, parents_indices=None,disk=0, duration=1):
        self.instance_id = instance_id
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.duration = duration
        self.parents = parents_indices


class TaskConfig(object):
    def __init__(self, task_index, task_instance_configs, parent_indices=None):
        self.task_index = task_index
        self.instances_number = len(task_instance_configs)
        if len(task_instance_configs) > 0:
            self.parent_indices = task_instance_configs[0].parents
        else:
            self.parent_indices = parent_indices
        self.task_instances = task_instance_configs

class JobConfig(object):
    def __init__(self, idx, submit_time, task_configs):
        self.submit_time = submit_time
        self.task_configs = task_configs
        self.id = idx
