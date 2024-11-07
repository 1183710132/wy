import os
import time
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CloudSimPy.core.machine import MachineConfig
from pgnet import *
from CloudSimPy.playground.DAG.utils.csv_reader_v2 import CSVReader
from CloudSimPy.playground.DAG.utils.feature_functions import features_extract_func_ac, features_normalize_func_ac
from CloudSimPy.playground.DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver
from CloudSimPy.playground.DAG.adapter.episode import Episode
from pgnet import PGnet, Agent


# 初始化仿真数据
machine_num = 3
job_num = 30
train_iter = 20

# 初始化奖励函数
reward_giver = MakespanRewardGiver(-1)
hidden_dim = 32
input_dim = 14

model_path = os.getcwd()+'/RL/model/{}'.format('PGnetV3')

# 要注意如果机器的cpu或者memory不够，就永远不会训练结束
machine_config = [MachineConfig(8, 1024, 1, mips=2, ) for i in range(machine_num)]

def getTracjectory(trajectory):
    observations = []
    actions = []
    actions_prob = []
    rewards = []

    for node in trajectory:
        observations.append(node.observation)
        actions.append(node.action)
        actions_prob.append(node.action_prob)
        rewards.append(node.reward)
    return observations, actions, actions_prob, rewards

def train(agent, job_data):
    loss_list = []
    csv_reader = CSVReader(job_data)
    job_config = csv_reader.generate(0, job_num)
    for step in range(train_iter):
        print('**************************** Iteration %i ***********************' % step)
        all_observations = []
        all_actions = []
        all_rewards = []
        # 时间戳
        makespans = []
        average_completions = []
        average_slowdowns = []
        trajectories = []

        tic = time.time()

        simulation_num = 5
        # 每step使用模拟simulation_num遍
        for i in range(simulation_num):
            algorithm = PGnet(agent, reward_giver, features_extract_func=features_extract_func_ac, 
                            features_normalize_func=features_normalize_func_ac)
            episode = Episode(machine_config, job_config, algorithm, None)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()
            trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
            makespans.append(episode.simulation.env.now)
        toc = time.time()
        
        for traj in trajectories:
            observations = []
            actions = []
            actions_prob = []
            rewards = []

            observations, actions, actions_prob, rewards = getTracjectory(traj)
            
            all_observations.append(observations)
            all_actions.append(actions_prob)
            all_rewards.append(rewards)

        all_q_s, all_adv = agent.estimate_return(all_rewards)
        agent.update_parameters(all_actions, all_q_s, all_adv)
        print('loss: ', agent.loss)
        loss_list.append(agent.loss)
    return loss_list
    # if not os.path.isdir(model_path):
    #     os.makedirs(model_path)
    # agent.save(model_path + '/model.pth')
    # plt.figure()
    # plt.plot(range(train_iter), loss_list)
    # plt.show()

def test(agent, job_data):
    csv_reader = CSVReader(job_data)
    job_config = csv_reader.generate(0, job_num)
    if os.path.isdir(model_path):
        agent.load(model_path + '/model.pth')
    algorithm = PGnet(agent, reward_giver, features_extract_func=features_extract_func_ac, 
                            features_normalize_func=features_normalize_func_ac)
    episode = Episode(machine_config, job_config, algorithm, None)
    algorithm.reward_giver.attach(episode.simulation)
    episode.run()
    scheduleflow = episode.simulation.scheduler.algorithm.scheduleflow
    for flow in scheduleflow:
        machine_id = flow[0].id
        task_id = flow[1].task_index
        clock = flow[2]
        print('task:{} run in machine:{} in clock {}'.format(task_id, machine_id, clock))
    # observations, actions, actions_prob, rewards = getTracjectory(trajectory)
    print()

def read_data():
    agent = Agent(hidden_dim, input_dim)
    task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_num = 30
    loss_list = []
    for t in task_type:
        for i in range(10):
            job_data = '{}\\RL\\csv_2\\{}_{}_{}.csv'.format(os.getcwd(), t, task_num, i)
            loss_list += train(agent, job_data)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()

if __name__ == '__main__':
    read_data()
    # test()