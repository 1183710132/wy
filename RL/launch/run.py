import os
import time
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt
import math
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CloudSimPy.core.machine import MachineConfig

from CloudSimPy.playground.DAG.utils.csv_reader import CSVReader
from CloudSimPy.playground.DAG.utils.feature_functions import features_extract_func
from CloudSimPy.playground.DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver, ComputePriceRewardGiver
from CloudSimPy.playground.DAG.adapter.episode import Episode
from a2c.A2C import Actor_Critic, Agent
from DRL.DRL import DRL, Node

# 初始化仿真数据
machine_num = 2
job_num = 30
train_iter = 20

# 初始化奖励函数
reward_giver = MakespanRewardGiver(-1)
hidden_dim = 128
input_dim = 22

model_path = os.getcwd()+'/model/{}'.format('A2CV1')

# 要注意如果机器的cpu或者memory不够，就永远不会训练结束
machine_config = [MachineConfig(2, 2048, 1, mips=20*math.pow(4, i), price=1*(i+1)) for i in range(machine_num)]

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


def train(agent, job_data, max_step=1000, temperature=1):
    loss_list = []
    csv_reader = CSVReader(job_data, machine_config)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    min_span = max_step
    for step in range(train_iter):
        print('**************************** Iteration %i ***********************' % step)
        all_observations = []
        all_actions = []
        all_actions_logpro = []
        all_rewards = []
        all_adv = []
        # 时间戳
        trajectories = []
        scheduleflows = []
        makespans = []
        tic = time.time()

        # 每step使用模拟simulation_num遍
        simulation_num = 10
        for i in range(simulation_num):
            algorithm = DRL(agent, reward_giver)
            episode = Episode(machine_config, job_config, algorithm, None)
            episode.simulation.cluster.task_instance_features = task_instance_features
            algorithm.reward_giver.attach(episode.simulation)
            episode.run(max_step=1000, temperature=temperature)
            trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
            scheduleflows.append(episode.simulation.scheduler.algorithm.scheduleflow)
            min_span = min(episode.simulation.env.now, min_span)
            t = episode.simulation.env.now
            makespans.append(t)
            print(t)
        toc = time.time()

        min_step = min_span

        for traj, flow, clock in zip(trajectories, scheduleflows, makespans):
            
            min_step = min(len(traj), min_step)

            observations, actions, actions_prob, actions_logpro, rewards, advs = getTracjectory(traj, clock, max_step=2*min_step)
            
            all_observations.append(observations)
            all_actions.append(actions_prob)
            all_actions_logpro.append(actions_logpro)
            all_rewards.append(rewards)
            all_adv.append(advs)

        agent.update_parameters(all_observations, all_actions, all_actions_logpro, all_rewards, all_adv)
        print('loss: ', agent.loss)
        loss_list.append(agent.loss)
    
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    agent.save(model_path + '/model.pth')
    return loss_list, makespans
    # if not os.path.isdir(model_path):
    #     os.makedirs(model_path)
    # agent.save(model_path + '/model.pth')
    # plt.figure()
    # plt.plot(range(train_iter), loss_list)
    # plt.show()

def test(agent, job_data, temperature=1):
    csv_reader = CSVReader(job_data, machine_config)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    if os.path.isdir(model_path):
        if os.path.exists(model_path + '/model.pth'):
            agent.load(model_path + '/model.pth')
    algorithm = DRL(agent, reward_giver)
    episode = Episode(machine_config, job_config, algorithm, None)
    episode.simulation.cluster.task_instance_features = task_instance_features
    algorithm.reward_giver.attach(episode.simulation)
    episode.run(temperature=temperature)
    scheduleflow = episode.simulation.scheduler.algorithm.scheduleflow
    prices = 0
    print(episode.simulation.env.now)
    for flow in scheduleflow:
        if flow[0] is None:
            prices += flow[2]
            continue
        machine_id = flow[0].id
        task_id = flow[1].task_index
        clock = flow[3]
        print('task:{} run in machine:{} in clock {} total cost:{}'.format(task_id, machine_id, clock, prices))
    
    return min([len(scheduleflow), 500])
    # observations, actions, actions_prob, rewards = getTracjectory(trajectory)

def read_data():
    agent = Agent(hidden_dim, input_dim)
    task_type = ['CyberShake', 'Genome', 'Montage', 'SIPHT']
    task_type = ['CyberShake']
    task_num = 10
    loss_list = []
    for t in task_type:
        for i in range(9):
            job_data = '{}/csv_2/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, task_num, i)
            print('start in data : ', t, i)
            print('----------------------before learning-----------------')
            test(agent, job_data)
            loss_list, _= train(agent, job_data)
            print('----------------------after leanring------------------')
            test(agent, job_data)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    job_data = '{}/csv_2/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, task_num, 9)
    agent.save(model_path + '/model.pth')
    test(agent, job_data)


def test_train():
    agent = Agent(hidden_dim, input_dim)
    # job_file = os.getcwd() + '/jobs_files/job_1.csv'
    job_file = os.getcwd() + '/csv_2/10/CyberShake_10_1.csv'
    max_step = 1000
    for i in range(10):
        step = test(agent, job_file, temperature=5)
        max_step = min(step, max_step)
    loss_list, makespan = train(agent, job_file, max_step=max_step, temperature=5)
    plt.figure()
    # plt.plot(range(len(loss_list)), loss_list)
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    for i in range(10):
        test(agent, job_file, temperature=5)

if __name__ == '__main__':
    # read_data()
    # test()
    # 示例：设置随机种子
    # seed = 42
    # set_seed(seed)
    test_train()