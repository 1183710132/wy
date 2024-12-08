import numpy as np
import torch
import torch.nn as nn
import copy

class Node(object):

    def __init__(self, observation, action, prob, logpro, reward, adv, clock, loss=None):
        # 定义观测状态，动作空间，奖励和时钟
        self.observation = observation
        self.action = action
        self.action_prob = prob
        self.action_logpro = logpro
        self.reward = reward
        self.adv = adv
        self.clock = clock
        self.loss = loss

class DRLPretrain(object):

    def __init__(self, agent, reward_giver):
        self.agent = agent
        self.reward_giver = reward_giver
        self.actions = []
        self.reward = []
        self.current_trajectory = []
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.scheduleflow = []
        self.rankScore = 1
        self.enhanced_feature = None

    def extract_features(self, valid_pairs, task_feature, all_tasks):
        features = []
        task_indexs = list(all_tasks.keys())
        for machine, task in valid_pairs:
            task_index = task_indexs.index(task.task_index)
            # 特征构成是由机器的cpu，机器内存和task的特征来构成的
            machine_feature = np.array([machine.cpu, machine.memory/1024, machine.mips/10, machine.price, task.task_config.memory/machine.mips])
            machine_feature = torch.tensor(machine_feature, dtype=torch.float).to(self.device)
            features.append(torch.cat((machine_feature, task_feature[task_index]), dim=0))
        return features
    
    def features_extract_func(self, task, task_instance_features):
        return [task.task_config.cpu, (task.task_config.memory-task_instance_features[3])/(task_instance_features[2]-task_instance_features[3]),
                task.EST/task.task_config.deadline, task.EFT/task.task_config.deadline, task.LFT/1000,
                task.feature['children_task_num'], task.feature['layer_task_num'],
                task.feature['task_memory_percent'], task.feature['parents_job_num']]
    
    def global_features(self, cluster, clock):
        # 全局特征：当前集群情况，有多少机器空闲，有多少机器运行中，有多少task在排队
        # features = [cluster.cpu_capacity, cluster.memory_capacity/1024, cluster.cpu, cluster.memory/1024, cluster.mips, len(cluster.running_task_instances), 
        #             len(cluster.finished_tasks), len(cluster.ready_unfinished_tasks), len(cluster.jobs), len(cluster.finished_jobs), len(cluster.unfinished_jobs)]
        features = [cluster.cpu, cluster.memory/1024, cluster.mips, len(cluster.running_task_instances), 
                    len(cluster.finished_tasks), len(cluster.ready_unfinished_tasks), len(cluster.unfinished_tasks), 0]
        return features

    def __call__(self, cluster, clock, temperature=1, is_cache=False):
        if clock == 0:
            return None, None
        # 先从集群获取所有的机器和等待调度的任务
        machines = cluster.machines[:len(cluster.machines)//2]
        waiting_tasks = cluster.ready_tasks_which_has_waiting_instance
        all_candidates = []
        task_set = []
        edge_index = cluster.edge_index
        all_tasks = list(cluster.all_tasks.values())

        # all task feature
        if self.enhanced_feature is None or not is_cache:
            # 缓存图嵌入
            all_task_feature = [self.features_extract_func(task, cluster.task_instance_features) for task in all_tasks]
            all_task_feature = torch.tensor(np.array(all_task_feature), dtype=torch.float).to(self.device) # [n, 9]
            edge_index = torch.squeeze(torch.tensor(np.array(edge_index)).to(self.device))
            self.enhanced_feature = self.agent.gcn(all_task_feature, edge_index)

        enhanced_feature = self.enhanced_feature

        ground_true = 10000
        for machine in machines:
            for task in waiting_tasks:
                # 判断是否满足调度条件，满足则加入到候选集中
                if machine.accommodate(task) and task.ready:
                    all_candidates.append((machine, task))
                    match = True if task.task_config.vmid==machine.id else False
                    if match:
                        if self.rankScore == task.rankScore:
                            ground_true = len(all_candidates)-1
                    task_set.append([task.task_index, task.task_config.vmid, machine.id, task.rankScore])

        if len(all_candidates) > 0:
            features = self.extract_features(all_candidates, enhanced_feature, cluster.all_tasks)
            features = torch.vstack(features)
            pass_feature = torch.zeros(features.shape[1]).to(self.device)
            # 新增一行，这一行的作用是pass轮空
            features = torch.vstack([features, pass_feature])
        else:
            features = torch.zeros([1, 13]).to(self.device)
        global_features = torch.tile(torch.tensor(self.global_features(cluster, clock), dtype=torch.float).to(self.device), (features.shape[0], 1)) # [n, 8]
        features = torch.hstack((features, global_features)) # [n, 21]
        # 选择动作概率密度
        action_logits, values = self.agent.select_action(features, temperature=temperature)
        
        if ground_true > len(all_candidates):
            ground_true = len(all_candidates)

        action = action_logits[ground_true]
        if ground_true == len(all_candidates):
            reward = self.reward_giver.get_reward()
            
            self.current_trajectory.append(Node(features, action, action, torch.log(action), reward, values[ground_true], clock, loss=torch.tensor(0).to(self.device)))
            self.scheduleflow.append((None, None, reward, clock))
            return None, None

        # target_machine = all_candidates[action_item][0]
        # target_task = all_candidates[action_item][1]
        # print('machine:{}, task:{}, clock:{}'.format(target_machine.id, target_task.task_index, clock))
        loss = -torch.log(action_logits[ground_true] + 1e-8)

        self.rankScore += 1

        node = Node(features, action, action, torch.log(action), 0, values[ground_true], clock, loss=loss)
        self.current_trajectory.append(node)
        self.scheduleflow.append((all_candidates[ground_true][0], all_candidates[ground_true][1], 0, clock))
        # print(all_candidates[ground_true][0].id, all_candidates[ground_true][1].task_index)
        return all_candidates[ground_true]