import os
import time
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CloudSimPy.core.machine import MachineConfig
from pgnet import *
from CloudSimPy.playground.DAG.utils.csv_reader import CSVReader
from CloudSimPy.playground.DAG.utils.feature_functions import features_extract_func
from CloudSimPy.playground.DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver, ComputePriceRewardGiver
from CloudSimPy.playground.DAG.adapter.episode import Episode
from pgnet import PGnet, Agent


# 初始化仿真数据
machine_num = 2
job_num = 30
train_iter = 100

# 初始化奖励函数
reward_giver = MakespanRewardGiver(-1)
hidden_dim = 64
input_dim = 19

model_path = os.getcwd()+'/model/{}'.format('PGnetV3')

# 要注意如果机器的cpu或者memory不够，就永远不会训练结束
machine_config = [MachineConfig(1, 1024, 1, mips=2*math.pow(10, i), price=1*(i+1)) for i in range(machine_num)]


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


def getFlowPriceReward(scheduleflow):
    prices = []
    for flow in scheduleflow:
        machine = flow[0]
        task = flow[1]
        if machine is None:
            price = -1
        else:
            price = task.memory / machine.mips * machine.price
        prices.append(price)
    return prices



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
        trajectories = []
        scheduleflows = []
        tic = time.time()

        simulation_num = 5
        # 每step使用模拟simulation_num遍
        for i in range(simulation_num):
            algorithm = PGnet(agent, reward_giver, features_extract_func=features_extract_func, 
                            )
            episode = Episode(machine_config, job_config, algorithm, None)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()
            trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
            scheduleflows.append(episode.simulation.scheduler.algorithm.scheduleflow)
            makespans.append(episode.simulation.env.now)
        toc = time.time()
        
        for traj, flow in zip(trajectories, scheduleflows):
            observations = []
            actions = []
            actions_prob = []
            rewards = []

            observations, actions, actions_prob, rewards = getTracjectory(traj)
            
            
            all_observations.append(observations)
            all_actions.append(actions_prob)
            all_rewards.append(rewards)

        all_q_s, all_adv = agent.estimate_return(all_actions, all_rewards)
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
    # if os.path.isdir(model_path):
    #     agent.load(model_path + '/model.pth')
    algorithm = PGnet(agent, reward_giver, features_extract_func=features_extract_func, 
                            )
    episode = Episode(machine_config, job_config, algorithm, None)
    algorithm.reward_giver.attach(episode.simulation)
    episode.run()
    scheduleflow = episode.simulation.scheduler.algorithm.scheduleflow
    prices = 0
    for flow in scheduleflow:
        if flow[0] is None:
            prices += flow[2]
            continue
        machine_id = flow[0].id
        task_id = flow[1].task_index
        clock = flow[3]
        print('task:{} run in machine:{} in clock {} total cost:{}'.format(task_id, machine_id, clock, prices))
    # observations, actions, actions_prob, rewards = getTracjectory(trajectory)

def read_data():
    agent = Agent(hidden_dim, input_dim)
    task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_type = ['CyberShake']
    task_num = 10
    loss_list = []
    for t in task_type:
        for i in range(1):
            job_data = '{}/csv_2/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, task_num, i)
            print('start in data : ', t, i)
            print('----------------------before learning-----------------')
            test(agent, job_data)
            loss_list += train(agent, job_data)
            print('----------------------after leanring------------------')
            test(agent, job_data)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    agent.save(model_path + '/model.pth')
    test(agent, job_data)


def test_train():
    agent = Agent(hidden_dim, input_dim)
    job_file = os.getcwd() + '/jobs_files/job_1.csv'
    test(agent, job_file)
    loss_list = train(agent, job_file)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    test(agent, job_file)

if __name__ == '__main__':
    # read_data()
    # test()
    test_train()