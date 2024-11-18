import numpy as np
import torch
import torch.nn as nn
import copy

class Node(object):

    def __init__(self, observation, action, prob, logpro, reward, adv, clock):
        # 定义观测状态，动作空间，奖励和时钟
        self.observation = observation
        self.action = action
        self.action_prob = prob
        self.action_logpro = logpro
        self.reward = reward
        self.adv = adv
        self.clock = clock
        self.loss = None

class DRL(object):

    def __init__(self, agent, reward_giver):
        self.agent = agent
        self.reward_giver = reward_giver
        self.actions = []
        self.reward = []
        self.current_trajectory = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.scheduleflow = []

    def extract_features(self, valid_pairs, task_instance_features):
        features = []
        for machine, task in valid_pairs:
            # 特征构成是由机器的cpu，机器内存和task的特征来构成的
            features.append([machine.cpu, machine.memory/1024, machine.mips/10, machine.price, task.task_config.memory/machine.mips] + self.features_extract_func(task, task_instance_features))
        return features
    
    def features_extract_func(self, task, task_instance_features):
        return [task.task_config.cpu, (task.task_config.memory-task_instance_features[3])/(task_instance_features[2]-task_instance_features[3]),
                task.EST/10, task.EFT/10, task.LFT/10,
                task.feature['first_layer_task'], task.feature['first_layer_instance'],
                task.feature['layers_task'], task.feature['child_task_numbers']]
    
    def global_features(self, cluster, clock):
        # 全局特征：当前集群情况，有多少机器空闲，有多少机器运行中，有多少task在排队
        # features = [cluster.cpu_capacity, cluster.memory_capacity/1024, cluster.cpu, cluster.memory/1024, cluster.mips, len(cluster.running_task_instances), 
        #             len(cluster.finished_tasks), len(cluster.ready_unfinished_tasks), len(cluster.jobs), len(cluster.finished_jobs), len(cluster.unfinished_jobs)]
        features = [cluster.cpu, cluster.memory/1024, cluster.mips, len(cluster.running_task_instances), 
                    len(cluster.finished_tasks), len(cluster.ready_unfinished_tasks), len(cluster.unfinished_tasks), 0]
        return features

    def __call__(self, cluster, clock, temperature=1):
        # 先从集群获取所有的机器和等待调度的任务
        machines = cluster.machines
        tasks = cluster.ready_tasks_which_has_waiting_instance
        all_candidates = []
        task_set = []
        for machine in machines:
            for task in tasks:
                # 判断是否满足调度条件，满足则加入到候选集中
                if machine.accommodate(task) and task.ready:
                    all_candidates.append((machine, task))
                    task_set.append((task.task_index, task.ready))
        # print(len(all_candidates), clock)
        # if len(all_candidates) == 0:
        #     reward = self.reward_giver.get_reward()
        #     self.current_trajectory.append(Node(None, None, None, None, reward, None, clock))
        #     self.scheduleflow.append((None, None, reward,clock))
        #     return None, None
        # else:
            # 提取候选集特征，转成tensor
        if len(all_candidates) > 0:
            features = self.extract_features(all_candidates, cluster.task_instance_features)
            pass_feature = np.zeros(14)
            # 新增一行，这一行的作用是pass轮空
            features = np.vstack([features, pass_feature])
        else:
            features = np.zeros([1, 14])
        global_features = np.tile(self.global_features(cluster, clock), (features.shape[0], 1))
        features = np.hstack((features, global_features))
        features = torch.tensor(np.array(features), dtype=torch.float).to(self.device)
        # 选择动作概率密度
        action_logits, values = self.agent.select_action(features, temperature=temperature)
        action_distribution = torch.distributions.Categorical(action_logits)
        # 采样动作
        action = action_distribution.sample()
        action_item = action.item()
        
        if action_item == len(all_candidates):
            reward = self.reward_giver.get_reward()
            
            self.current_trajectory.append(Node(features, action, action_logits[action_item], action_distribution.log_prob(action), reward, values[action_item], clock))
            self.scheduleflow.append((None, None, reward, clock))
            return None, None
        
        # target_machine = all_candidates[action_item][0]
        # target_task = all_candidates[action_item][1]
        # print('machine:{}, task:{}, clock:{}'.format(target_machine.id, target_task.task_index, clock))

        node = Node(features, action, action_logits[action_item], action_distribution.log_prob(action), 0, values[action_item], clock)
        self.current_trajectory.append(node)
        self.scheduleflow.append((all_candidates[action_item][0], all_candidates[action_item][1], 0, clock))
        return all_candidates[action_item]