import numpy as np
import torch
import torch.nn as nn
import copy

class PolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x, temperature=1):
        x = self.layer(x)
        # 输出的是概率，所以要做标准化
        x = torch.softmax(x/temperature, dim=0)
        return x

class Node(object):

    def __init__(self, observation, action, prob, logpro, reward, clock):
        # 定义观测状态，动作空间，奖励和时钟
        self.observation = observation
        self.action = action
        self.action_prob = prob
        self.action_logpro = logpro
        self.reward = reward
        self.clock = clock
        self.loss = None

class Agent(object):

    def __init__(self, hidden_dim, input_dim, lr=0.01, gamma=0.98):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.network = PolicyNet(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, observation, temperature=1):
        logits = self.network.forward(observation, temperature=temperature)
        # 这里有个bug，如果只有一个候选时，直接squeeze shape=[]，应该保留一维
        return torch.squeeze(logits, dim=-1)

    def _sum_of_rewards(self, actions_n, rewards_n, reward_to_go=True):
        q_s = []
        for ac, re in zip(actions_n, rewards_n):
            q = []
            cur_q = 0
            for action, reward in zip(reversed(ac), reversed(re)):
                if action is None:
                    cur_q = cur_q + reward
                else:
                    cur_q = cur_q * self.gamma + reward
                q.append(cur_q)
            q = list(reversed(q))
            q_s.append(q)
        # 判断是否要进行奖励衰减
        if reward_to_go:
            return q_s
        else:
            q_n = []
            for q in q_s:
                q_n.append([q[0]]*len(q))

    def _compute_advantage(self, q_n, baseline=True):
        # 使用baseline的目的是使用优势函数，稳定奖励的方差，使更新更加稳定
        if baseline:
            adv_n = copy.deepcopy(q_n)
            max_length = max([len(adv) for adv in adv_n])
            for adv in adv_n:
                while len(adv) < max_length:
                    adv.append(0)
            adv_n = np.array(adv_n)
            adv_n = adv_n - adv_n.mean(axis=0)
            adv_n_ = []
            for i in range(adv_n.shape[0]):
                original_length = len(q_n[i])
                adv_n_.append(list(adv_n[i][:original_length]))
                return adv_n_
        else:
            adv_n = q_n.copy()
            return adv_n

    def estimate_return(self, actions_n, rewards_n, normalize_advantage=True, reward_to_go=True, baseline=True):
        # 计算q值和v值，v值是优势函数，用于稳定策略学习
        q_n = self._sum_of_rewards(actions_n, rewards_n, reward_to_go=reward_to_go)
        adv_n = self._compute_advantage(rewards_n, baseline=baseline)

        if normalize_advantage:
            adv_s = sum(adv_n, [])
            adv_s = np.array(adv_s)
            mean = adv_s.mean()
            std = adv_s.std()
            adv_n_ = []
            for advantages in adv_n:
                adv_n_.append([(vn-mean)/(std+np.finfo(np.float32).eps) for vn in advantages])
            adv_n = adv_n_
        return q_n, adv_n
    
    def update_parameters(self, actions, actions_logpro, rewards, advantages, lamb=1):
        loss = []
        entropy = []
        for action, act_log, reward, adv in zip(actions, actions_logpro, rewards, advantages):
            l = []
            e = []
            for a, alp, r, v in zip(action, act_log, reward, adv):
                if a is None:
                    continue
                l.append(-alp * (r-v))
                e.append(-a * alp)
            loss.append(torch.stack(l))
            entropy.append(torch.stack(e))

        # 对模型进行熵正则化惩罚
        loss = torch.cat(loss, dim=0).mean(dim=0)
        entropy = torch.cat(entropy, dim=0).sum(dim=0)
        self.optimizer.zero_grad()
        loss = loss.mean() - lamb * entropy.mean()
        self.loss = loss.item()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path, weights_only=True):
        self.network.load_state_dict(torch.load(path, weights_only=weights_only))


class PGnet(object):

    def __init__(self, agent, reward_giver, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.actions = []
        self.reward = []
        self.current_trajectory = []
        self.features_extract_func = features_extract_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.scheduleflow = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, task in valid_pairs:
            # 特征构成是由机器的cpu，机器内存和task的特征来构成的
            features.append([machine.cpu, machine.memory/1024, machine.mips, machine.price, task.task_config.memory/machine.mips] + self.features_extract_func(task))
        return features
    
    def global_features(self, cluster):
        # 全局特征：当前集群情况，有多少机器空闲，有多少机器运行中，有多少task在排队
        features = [cluster.cpu_capacity, cluster.memory_capacity/1024, cluster.cpu, cluster.memory/1024, cluster.mips, len(cluster.running_task_instances), 
                    len(cluster.finished_tasks), len(cluster.ready_unfinished_tasks), len(cluster.jobs), len(cluster.finished_jobs), len(cluster.unfinished_jobs)]
        # features = [cluster.cpu_capacity, cluster.memory_capacity/1024, cluster.cpu, cluster.memory/1024, cluster.mips, len(cluster.running_task_instances), 
        #             len(cluster.finished_tasks), len(cluster.ready_unfinished_tasks)]
        return features

    def __call__(self, cluster, clock):
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
        if len(all_candidates) == 0:
            reward = self.reward_giver.get_reward()
            self.current_trajectory.append(Node(None, None, None, None, reward, clock))
            self.scheduleflow.append((None, None, reward,clock))
            return None, None
        else:
            # 提取候选集特征，转成tensor
            features = self.extract_features(all_candidates)
            pass_feature = np.zeros(len(features[0]))
            # # 新增一行，这一行的作用是pass轮空
            features = np.vstack([features, pass_feature])
            global_features = np.tile(self.global_features(cluster), (features.shape[0], 1))
            features = np.hstack((features, global_features))
            features = torch.tensor(np.array(features), dtype=torch.float).to(self.device)
            # 选择动作概率密度
            action_logits = self.agent.select_action(features, temperature=1)
            action_distribution = torch.distributions.Categorical(action_logits)
            # 采样动作
            action = action_distribution.sample()
            action_item = action.item()
            
            if action_item == len(all_candidates):
                reward = self.reward_giver.get_reward()
                
                self.current_trajectory.append(Node(features, action, action_logits[action_item], action_distribution.log_prob(action), reward, clock))
                self.scheduleflow.append((None, None, reward, clock))
                return None, None
            
            # target_machine = all_candidates[action_item][0]
            # target_task = all_candidates[action_item][1]
            # print('machine:{}, task:{}, clock:{}'.format(target_machine.id, target_task.task_index, clock))
            

            node = Node(features, action, action_logits[action_item], action_distribution.log_prob(action), 0, clock)
            self.current_trajectory.append(node)
            self.scheduleflow.append((all_candidates[action_item][0], all_candidates[action_item][1], 0, clock))
        return all_candidates[action_item]