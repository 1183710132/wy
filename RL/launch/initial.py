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


# 初始化仿真数据
job_num = 10
file_num = 1000
is_gcn = False
is_a2c = True

input_dim = 22
if is_gcn:
    input_dim = 21
    from DRL.DRL_gat_pretrain import DRLPretrain, Node
    from DRL.DRL_gcn import DRL
else:
    input_dim = 22
    from DRL.DRL_pretrain import DRLPretrain, Node
    from DRL.DRL import DRL

# 初始化奖励函数
reward_giver = MakespanRewardGiver(-1)
hidden_dim = 64

model_file = 'a2c_file_{}_job_{}_task_10'.format(file_num, job_num)
model_path = os.getcwd()+'/model/pretrain/{}'.format(model_file)

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


def pretrain(agent, job_data, max_step=1000, temperature=1):
    loss_list = []
    csv_reader = CSVReaderPretrain(job_data, total_machine)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    min_span = max_step
    train_iter = 1
    for step in range(train_iter):
        # print('**************************** Iteration %i ***********************' % step)
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

        agent.pretrain(all_actions_logpro, all_rewards, all_adv, all_cross_loss , is_a2c=is_a2c)
        print('loss: ', agent.loss)
        loss_list.append(agent.loss)

    # if not os.path.isdir(model_path):
    #     os.makedirs(model_path)
    # agent.save(model_path + '/model.pth')
    return loss_list, makespans


def test(agent, job_data, temperature=1):
    csv_reader = CSVReaderPretrain(job_data, total_machine)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    makespan = 10000
    if os.path.isdir(model_path):
        if os.path.exists(model_path + '/model.pth'):
            agent.load(model_path + '/model.pth')
    for i in range(10):
        algorithm = DRL(agent, reward_giver)
        episode = Episode(total_machine, job_config, algorithm, None)
        episode.simulation.cluster.task_instance_features = task_instance_features
        episode.simulation.cluster.task_edge_index = csv_reader.task_edge_index
        algorithm.reward_giver.attach(episode.simulation)
        episode.run(temperature=temperature)
        scheduleflow = episode.simulation.scheduler.algorithm.scheduleflow
        prices = 0
        makespan = min(episode.simulation.env.now, makespan)
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
    
    train_num = (int)(0.9*file_num)
    test_makespan = []
    for t in task_type:
        for i in range(train_num):
            job_data = '{}/csv_pretrain_ceda/{}/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, t, task_num, i)
            print('start in data : ', t, i)
            # print('----------------------before learning-----------------')
            # test(agent, job_data)
            loss, _= pretrain(agent, job_data, temperature=5)
            loss_list += loss
            # print('----------------------after leanring------------------')
            # test(agent, job_data)
        for j in range(train_num+1, file_num):
            job_data = '{}/csv_pretrain_ceda/{}/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, t, task_num, j)
            print('start in data : ', t, j)
            test_makespan.append(test(agent, job_data, temperature=1))
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    smoothed_loss = pd.Series(loss_list).rolling(window=20).mean()
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

    photo_file = os.getcwd() + '/photo/' + model_file
    if not os.path.isdir(photo_file):
        os.makedirs(photo_file)
    plt.savefig( photo_file+ '/result.png')
    plt.show()

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    agent.save(model_path + '/model.pth')

if __name__ == '__main__':
    # read_data()
    # test()
    # 示例：设置随机种子
    # seed = 42
    # set_seed(seed)
    create_vm(machine_type, machine_num, total_machine)
    read_data()