import os
import time
import numpy as np
import pandas as pd
import torch
import sys
from matplotlib import pyplot as plt
import math
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CloudSimPy.core.machine import MachineConfig

from CloudSimPy.playground.DAG.utils.csv_reader import CSVReaderPretrain
from CloudSimPy.playground.DAG.utils.feature_functions import features_extract_func
from CloudSimPy.playground.DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver, ComputePriceRewardGiver
from CloudSimPy.playground.DAG.adapter.episode import Episode
from a2c.A2C import Actor_Critic, Agent
from DRL.DRL_pretrain import DRLPretrain, Node
from DRL.DRL import DRL

# 初始化仿真数据
machine_num = 2
job_num = 30


# 初始化奖励函数
reward_giver = MakespanRewardGiver(-1)
hidden_dim = 128
input_dim = 22

model_path = os.getcwd()+'/model/{}'.format('A2CV4')

# 要注意如果机器的cpu或者memory不够，就永远不会训练结束
machine_config = {'stay':[], 'rent':[]}
machine_type = {'F2':[2, 4, 0.192, 0.1539], 'F4':[4, 8, 0.383, 0.3099], 'F8':[8, 16, 0.766, 0.6158]}
machine_num = {'F2': [4, 4], 'F4':[4, 4], 'F8':[4, 4]}
total_machine = []

def create_vm(machine_type, machine_num, total_machine):
    for key, value in machine_num.items():
        m_type = machine_type[key]
        stay_machine = [MachineConfig(1, m_type[1]*1024, 1, mips=10*m_type[0], price=m_type[3]) for _ in range(value[0])]
        machine_config['stay'] += stay_machine
        total_machine += stay_machine
    
    for key, value in machine_num.items():
        m_type = machine_type[key]
        rent_machine = [MachineConfig(1, m_type[1]*1024, 1, mips=10*m_type[0], price=m_type[3]) for _ in range(value[1])]
        machine_config['rent'] += rent_machine
        total_machine += rent_machine

def set_seed(seed):
    """
    设置随机种子以保证实验可重复性
    """
    # 设置 Python 内置的随机数种子
    random.seed(seed)
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    # 设置 PyTorch 的随机数种子（CPU 和 GPU）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 设置 PyTorch 的确定性选项（影响卷积等操作）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getTracjectory(trajectory, clock, max_step=1000):
    observations = []
    actions = []
    actions_prob = []
    actions_logpro = []
    rewards = []
    advs = []
    step = len(trajectory)
    xita = 0
    if clock >= max_step:
        xita = 1000000
    for node in trajectory:
        if node.observation is None:
            continue
        observations.append(node.observation)
        actions.append(node.action)
        actions_prob.append(node.action_prob)
        actions_logpro.append(node.action_logpro)
        rewards.append(node.reward)
        advs.append(node.adv)
    rewards[-1] -= xita
    return observations, actions, actions_prob, actions_logpro, rewards, advs


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


def pretrain(agent, job_data, max_step=1000, temperature=1):
    loss_list = []
    csv_reader = CSVReaderPretrain(job_data, total_machine)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    min_span = max_step
    train_iter = 2
    for step in range(train_iter):
        print('**************************** Iteration %i ***********************' % step)
        all_observations = []
        all_actions = []
        all_actions_logpro = []
        all_rewards = []
        all_adv = []
        all_cross_loss = []
        # 时间戳
        trajectories = []
        makespans = []

        algorithm = DRLPretrain(agent, reward_giver)
        episode = Episode(total_machine, job_config, algorithm, None)
        episode.simulation.cluster.task_instance_features = task_instance_features
        episode.simulation.cluster.task_edge_index = csv_reader.task_edge_index
        algorithm.reward_giver.attach(episode.simulation)
        episode.run(max_step=1000, temperature=temperature)
        trajectories = episode.simulation.scheduler.algorithm.current_trajectory
        min_span = min(episode.simulation.env.now, min_span)
        t = episode.simulation.env.now
        makespans.append(t)

        for node in trajectories:
        
            all_observations.append(node.observation)
            all_actions.append(node.action_prob)
            all_actions_logpro.append(node.action_logpro)
            all_rewards.append(node.reward)
            all_adv.append(node.adv)
            all_cross_loss.append(node.loss)

        agent.pretrain(all_actions_logpro, all_rewards, all_adv, all_cross_loss)
        print('loss: ', agent.loss)
        loss_list.append(agent.loss)

    # if not os.path.isdir(model_path):
    #     os.makedirs(model_path)
    # agent.save(model_path + '/model.pth')
    return loss_list, makespans


def test(agent, job_data, temperature=1):
    csv_reader = CSVReaderPretrain(job_data, total_machine)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    if os.path.isdir(model_path):
        if os.path.exists(model_path + '/model.pth'):
            agent.load(model_path + '/model.pth')
    algorithm = DRL(agent, reward_giver)
    episode = Episode(total_machine, job_config, algorithm, None)
    episode.simulation.cluster.task_instance_features = task_instance_features
    episode.simulation.cluster.task_edge_index = csv_reader.task_edge_index
    algorithm.reward_giver.attach(episode.simulation)
    episode.run(temperature=temperature)
    scheduleflow = episode.simulation.scheduler.algorithm.scheduleflow
    prices = 0
    makespan = episode.simulation.env.now
    print(makespan)
    # for flow in scheduleflow:
    #     if flow[0] is None:
    #         prices += flow[2]
    #         continue
    #     machine_id = flow[0].id
    #     task_id = flow[1].task_index
    #     clock = flow[3]
    #     print('task:{} run in machine:{} in clock {} total cost:{}'.format(task_id, machine_id, clock, prices))
    
    return min([(makespan-csv_reader.deadline)/csv_reader.deadline, 500])
    # observations, actions, actions_prob, rewards = getTracjectory(trajectory)


def read_data():
    agent = Agent(hidden_dim, input_dim)
    task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_type = ['CyberShake']
    task_num = 10
    loss_list = []
    file_num = 1000
    train_num = 900
    test_makespan = []
    for t in task_type:
        for i in range(train_num):
            job_data = '{}/csv_pretrain_ceda/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, task_num, i)
            print('start in data : ', t, i)
            # print('----------------------before learning-----------------')
            # test(agent, job_data)
            loss, _= pretrain(agent, job_data, temperature=5)
            loss_list += loss
            # print('----------------------after leanring------------------')
            # test(agent, job_data)
        for j in range(file_num-train_num):
            job_data = '{}/csv_pretrain_ceda/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, task_num, j+train_num)
            print('start in data : ', t, j+train_num)
            test_makespan.append(test(agent, job_data, temperature=0.5))
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    smoothed_loss = pd.Series(loss_list).rolling(window=10).mean()
    smoothed_makespan = pd.Series(test_makespan).rolling(window=10).mean()

    axes[0, 0].plot(loss_list, label='tran_loss')
    axes[0, 0].legend()
    axes[1, 0].plot(test_makespan, label='test_makespan')
    axes[1, 0].legend()

    axes[0, 1].plot(smoothed_loss, label='smooth_loss')
    axes[0, 1].legend()
    axes[1, 1].plot(smoothed_makespan, label='smooth_makespan')
    axes[1, 1].legend()
    plt.tight_layout() # 自动调节子图间距
    plt.show()

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    agent.save(model_path + '/model.pth')

def test_train():
    agent = Agent(hidden_dim, input_dim)
    # job_file = os.getcwd() + '/jobs_files/job_1.csv'
    job_file = os.getcwd() + '/csv_pretrain_ceda/10/CyberShake_10_1.csv'
    max_step = 1000
    # for i in range(10):
    #     step = test(agent, job_file, temperature=5)
    #     max_step = min(step, max_step)
    loss_list, makespan = pretrain(agent, job_file, max_step=max_step, temperature=5)
    plt.figure()
    # plt.plot(range(len(loss_list)), loss_list)
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    # for i in range(10):
    test(agent, job_file, temperature=5)

if __name__ == '__main__':
    # read_data()
    # test()
    # 示例：设置随机种子
    # seed = 42
    # set_seed(seed)
    create_vm(machine_type, machine_num, total_machine)
    read_data()