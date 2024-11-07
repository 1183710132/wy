"""
    是对一次仿真的建模，
    一次仿真必须构造一个集群 Cluster 实例；
    构造一系列作业配置 JobConfig 实例，
    利用这些作业配置实例构造一个 Broker 实例；
    构造一个调度器 Scheduler 实例。
    在一次仿真可以选择开是否使用一个 Monitor 实例进行仿真过程的监测
"""


from core.monitor import Monitor


class Simulation(object):
    def __init__(self, env, cluster, task_broker, scheduler, event_file):
        self.env = env
        self.cluster = cluster
        self.task_broker = task_broker
        self.scheduler = scheduler
        self.event_file = event_file
        if event_file is not None:
            self.monitor = Monitor(self)

        # 注册到broker中，自动提交作业
        self.task_broker.attach(self)
        # 注册到调度器中
        self.scheduler.attach(self)

    def run(self):
        # Starting monitor process before task_broker process
        # and scheduler process is necessary for log records integrity.
        # 这里启动仿真过程，启动了监控进程，作业提交模块和调度器模块
        if self.event_file is not None:
            self.env.process(self.monitor.run())
        self.env.process(self.task_broker.run())
        self.env.process(self.scheduler.run())

    @property
    def finished(self):
        return self.task_broker.destroyed \
               and len(self.cluster.unfinished_jobs) == 0
