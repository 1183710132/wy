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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.layer(x)
        # 输出的是概率，所以要做标准化
        x = torch.softmax(x, dim=0)
        return x

class Node(object):

    def __init__(self, observation, action, prob, reward, clock):
        # 定义观测状态，动作空间，奖励和时钟
        self.observation = observation
        self.action = action
        self.action_prob = prob
        self.reward = reward
        self.clock = clock
        self.loss = None

class Agent(object):

    def __init__(self, hidden_dim, input_dim, lr=0.02, gamma=0.98):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.network = PolicyNet(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, observation):
        logits = self.network.forward(observation)
        # 这里有个bug，如果只有一个候选时，直接squeeze shape=[]，应该保留一维
        return torch.squeeze(logits, dim=-1)

    def _sum_of_rewards(self, rewards_n, reward_to_go=True):
        q_s = []
        for re in rewards_n:
            q = []
            cur_q = 0
            for reward in reversed(re):
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

    def estimate_return(self, rewards_n, normalize_advantage=True, reward_to_go=True, baseline=True):
        # 计算q值和v值，v值是优势函数，用于稳定策略学习
        q_n = self._sum_of_rewards(rewards_n, reward_to_go=reward_to_go)
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
    
    def update_parameters(self, actions, rewards, advantages):
        loss = []
        for action, reward, adv in zip(actions, rewards, advantages):
            l = []
            for a, r, v in zip(action, reward, adv):
                if a is None:
                    continue
                l.append(-a * (r-v))
            loss.append(torch.stack(l))
        loss = torch.cat(loss, dim=0).mean(dim=0)
        self.optimizer.zero_grad()
        loss = loss.mean()
        self.loss = loss.item()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path, weights_only=True):
        self.network.load_state_dict(torch.load(path, weights_only=weights_only))


class PGnet(object):

    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.actions = []
        self.reward = []
        self.current_trajectory = []
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.scheduleflow = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, task in valid_pairs:
            # 特征构成是由机器的cpu，机器内存和task的特征来构成的
            features.append([machine.cpu, machine.memory] + self.features_extract_func(task))
        features = self.features_normalize_func(features)
        return features
    
    def __call__(self, cluster, clock):
        # 先从集群获取所有的机器和等待调度的任务
        machines = cluster.machines
        tasks = cluster.ready_tasks_which_has_waiting_instance
        all_candidates = []
        for machine in machines:
            for task in tasks:
                # 判断是否满足调度条件，满足则加入到候选集中
                if machine.accommodate(task):
                    all_candidates.append((machine, task))
        # print(len(all_candidates), clock)
        if len(all_candidates) == 0:
            self.current_trajectory.append(Node(None, None, None, self.reward_giver.get_reward(), clock))
            return None, None
        else:
            # 提取候选集特征，转成tensor
            features = self.extract_features(all_candidates)
            features = torch.tensor(np.array(features), dtype=torch.float).to(self.device)
            # 选择动作概率密度
            action_distribution = self.agent.select_action(features)
            # 采样动作
            action = torch.multinomial(action_distribution, 1)
            action_item = action.item()
            node = Node(features, action, action_distribution[action_item], 0, clock)
            # target_machine = all_candidates[action_item][0]
            # target_task = all_candidates[action_item][1]
            # print('machine:{}, task:{}, clock:{}'.format(target_machine.id, target_task.task_index, clock))
            self.current_trajectory.append(node)
            self.scheduleflow.append((all_candidates[action_item][0], all_candidates[action_item][1], clock))
        return all_candidates[action_item]