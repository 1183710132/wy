"""
    是对于调度器的建模，通过策略模式这一设计模式，不同的 Scheduler 实例可以使用不同的调度算法进行调度
    调度器内置调度算法，决策过程就是通过算法来进行决策，所以调度算法要实现的是当前集群状态和时间下，将哪些task运行在哪些vm上
"""


class Scheduler(object):
    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.valid_pairs = {}

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def make_decision(self):
        while True:
            machine, task = self.algorithm(self.cluster, self.env.now)
            if machine is None or task is None:
                break
            else:
                task.start_task_instance(machine)

    def run(self, max_step=1000):
        while not self.simulation.finished:
            self.make_decision()
            yield self.env.timeout(1)
            if self.env.now > max_step:
                break
        self.destroyed = True
