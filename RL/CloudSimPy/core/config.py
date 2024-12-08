class TaskInstanceConfig(object):
    def __init__(self, task_config):
        self.cpu = task_config.cpu
        self.memory = task_config.memory
        self.disk = task_config.disk
        self.duration = task_config.duration


class TaskConfig(object):
    def __init__(self, serise, deadline, pretrain=False):
        self.task_index = serise.task_id
        self.instances_number = serise.instances_num
        self.cpu = serise.cpu
        self.memory = serise.memory
        self.disk = serise.disk
        self.duration = serise.duration
        self.parent_indices = serise.parents
        self.children = serise.children
        self.EST = serise.initial_EST
        self.EFT = serise.initial_EFT
        self.LFT = serise.initial_LFT - 1000 + deadline
        self.MET = serise.initial_MET
        self.deadline = deadline
        if pretrain:
            self.vmid = serise.VmId
            self.rankScore = serise.rankScore


class JobConfig(object):
    def __init__(self, idx, submit_time, task_configs, parent_indices=None):
        self.submit_time = submit_time
        self.task_configs = task_configs
        self.id = idx
        if len(task_configs) < 1:
            parent_indices = []
        else:
            parent_indices = task_configs[0].parent_indices
        self.parent_indices = parent_indices

