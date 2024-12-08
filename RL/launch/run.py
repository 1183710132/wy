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

from CloudSimPy.playground.DAG.utils.csv_reader import CSVReaderPretrain
from CloudSimPy.playground.DAG.utils.feature_functions import features_extract_func
from CloudSimPy.playground.DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver, ComputePriceRewardGiver
from CloudSimPy.playground.DAG.adapter.episode import Episode
from a2c.A2C import Actor_Critic, Agent
from DRL.DRL import DRL, Node

is_gcn = False

if not is_gcn:
    from DRL.DRL_pretrain import DRLPretrain
    from DRL.DRL import DRL
    input_dim = 22
else:
    input_dim = 21
    from DRL.DRL_gcn import DRL
    from DRL.DRL_gat_pretrain import DRLPretrain

# 初始化仿真数据
job_num = 10
is_a2c = True
train_iter = 50

# 初始化奖励函数
# reward_giver = ComputePriceRewardGiver(-1)
reward_giver = MakespanRewardGiver(-1)
hidden_dim = 64

read_model_path = os.getcwd()+'/model/pretrain/{}'.format('a2c_file_5000_job_10_task_10')
save_model_path = os.getcwd()+'/model/train/{}'.format('a2c_file_5000_job_10_task_10')

machine_config = {'stay':[], 'rent':[]}
machine_type = {'F2':[2, 4, 0.192, 0.1539], 'F4':[4, 8, 0.383, 0.3099], 'F8':[8, 16, 0.766, 0.6158]}
machine_num = {'F2': [4, 4], 'F4':[4, 4], 'F8':[4, 4]}
total_machine = []

# 要注意如果机器的cpu或者memory不够，就永远不会训练结束
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
    xita = clock
    if clock >= max_step:
        xita = 10000
    for node in trajectory:
        if node.observation is None:
            continue
        observations.append(node.observation)
        actions.append(node.action)
        actions_prob.append(node.action_prob)
        actions_logpro.append(node.action_logpro)
        rewards.append(node.reward)
        advs.append(node.adv)
    # rewards[-1] -= xita*xita
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


def train(agent, job_data, max_step=1000, temperature=1, deadline_xita=1):
    loss_list = []
    csv_reader = CSVReaderPretrain(job_data, total_machine)
    deadline = csv_reader.deadline * deadline_xita
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
            episode = Episode(total_machine, job_config, algorithm, None)
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

        for traj, flow, clock in zip(trajectories, scheduleflows, makespans):

            observations, actions, actions_prob, actions_logpro, rewards, advs = getTracjectory(traj, clock, max_step=1.5*deadline)
            
            all_observations.append(observations)
            all_actions.append(actions_prob)
            all_actions_logpro.append(actions_logpro)
            all_rewards.append(rewards)
            all_adv.append(advs)

        agent.update_parameters(all_observations, all_actions, all_actions_logpro, all_rewards, all_adv, is_a2c=is_a2c)
        print('loss: ', agent.loss)
        loss_list.append(agent.loss)
    
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    agent.save(save_model_path + '/model.pth')
    return loss_list, makespans

def test(agent, job_data, temperature=1):
    csv_reader = CSVReaderPretrain(job_data, total_machine)
    job_config, task_instance_features = csv_reader.generate(0, job_num)
    if os.path.isdir(read_model_path):
        if os.path.exists(read_model_path + '/model.pth'):
            agent.load(read_model_path + '/model.pth')
    algorithm = DRL(agent, reward_giver)
    episode = Episode(total_machine, job_config, algorithm, None)
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
            job_data = '{}/csv_pretrain_ceda/{}/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, t, task_num, i)
            print('start in data : ', t, i)
            print('----------------------before learning-----------------')
            test(agent, job_data)
            loss, _= train(agent, job_data)
            loss_list += loss
            print('----------------------after leanring------------------')
            test(agent, job_data)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    job_data = '{}/csv_pretrain_ceda/{}/{}/{}_{}_{}.csv'.format(os.getcwd(), task_num, t, t, task_num, 9)
    agent.save(save_model_path + '/model.pth')
    test(agent, job_data)


def test_train():
    agent = Agent(hidden_dim, input_dim)
    agent.load(read_model_path + '/model.pth')
    # job_file = os.getcwd() + '/jobs_files/job_1.csv'
    job_file = os.getcwd() + '/csv_pretrain_ceda/10/Cybershake/CyberShake_10_0.csv'
    # /home/wy/project/wy/RL/csv_pretrain_ceda/10/Cybershake/CyberShake_10_0.csv
    max_step = 1000
    for i in range(10):
        step = test(agent, job_file, temperature=1)
        max_step = min(step, max_step)
    loss_list, makespan = train(agent, job_file, max_step=max_step, temperature=1)
    plt.figure()
    # plt.plot(range(len(loss_list)), loss_list)
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
    for i in range(10):
        test(agent, job_file, temperature=1)

if __name__ == '__main__':
    create_vm(machine_type, machine_num, total_machine)
    # read_data()
    # test()
    # 示例：设置随机种子
    # seed = 42
    # set_seed(seed)
    test_train()