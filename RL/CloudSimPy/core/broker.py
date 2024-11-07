"""
    Broker 代替用户对计算集群提交作业
"""

from core.job import Job


class Broker(object):
    job_cls = Job

    def __init__(self, env, job_configs):
        self.env = env
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.job_configs = job_configs

    def attach(self, simulation):
        # 注册集群和仿真过程
        self.simulation = simulation
        self.cluster = simulation.cluster

    def run(self):
        for job_config in self.job_configs:
            assert job_config.submit_time >= self.env.now
            # 这里进程会暂停，直到submit_time >= now时才会继续仿真
            yield self.env.timeout(job_config.submit_time - self.env.now)
            # 此时进程会创建job，然后将job加入到集群中
            job = Broker.job_cls(self.env, job_config)
            # print('a task arrived at time %f' % self.env.now)
            self.cluster.add_job(job)
            job.attach(self.cluster)
        self.destroyed = True
