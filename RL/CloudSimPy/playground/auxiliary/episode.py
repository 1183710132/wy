import simpy

import sys
sys.path.append('CloudSimPy/')

from CloudSimPy.core.cluster import Cluster
from CloudSimPy.core.scheduler import Scheduler
from CloudSimPy.core.broker import Broker
from CloudSimPy.core.simulation import Simulation


class Episode(object):
    broker_cls = Broker

    def __init__(self, machine_configs, task_configs, algorithm, event_file):
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(machine_configs)

        task_broker = Episode.broker_cls(self.env, task_configs)

        scheduler = Scheduler(self.env, algorithm)

        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file)

    def run(self):
        self.simulation.run()
        self.env.run()
