import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import GCNConv


class Actor_Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, negative_slope=0.01):
        super(Actor_Critic, self).__init__()
        self.actor_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_dim, output_dim)
        )
        self.critic_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_dim, 1)
        )
        self.negative_slope = negative_slope  # 用于初始化
        self._initialize_weights()  # 初始化权重和偏置

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用适配 Leaky ReLU 的 He 初始化
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='leaky_relu', 
                    a=self.negative_slope
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, t=1):
        actor = self.actor_layer(x)
        actor = torch.softmax(actor/t, dim=0)
        value = self.critic_layer(x)
        return actor, value
    
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


class WorkflowGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WorkflowGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = torch.relu(x)  # 激活函数
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        return x

class Agent(object):

    def __init__(self, hidden_dim, input_dim, lr=0.01, gamma=0.99, output_dim=16, is_gcn=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.gcn = WorkflowGCN(self.input_dim, self.hidden_dim, output_dim)
        self.network = Actor_Critic(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.network.actor_layer.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.network.critic_layer.parameters(), lr=2*lr)
        self.is_gcn = is_gcn

    def select_action(self, observation, temperature=1):
        logits, values = self.network.forward(observation, t=temperature)
        # 这里有个bug，如果只有一个候选时，直接squeeze shape=[]，应该保留一维
        return torch.squeeze(logits, dim=-1), torch.squeeze(values, dim=-1)
    

    def _compute_adv(self, actions, rewards):
        res = []
        rt = 0
        sum_ = []
        
        for action, reward in zip(actions, rewards):
            l = []
            rt = 0
            godown = True
            for a, r in zip(reversed(action),reversed(reward)):
                # 削弱极端错误动作由于时间拉长对奖励传递的影响, 确保惩罚对该轨迹上的动作都生效
                if a > 1-1e-5 or r < -10*len(reward):
                    godown = False
                if not godown:
                    rt = rt + r
                else:
                    rt = rt * self.gamma + r
                l.append(rt)
                sum_.append(rt)
            res.append(list(reversed(l)))
        re_max = np.max(sum_)
        re_min = np.min(sum_)
        return res, re_max, re_min

    def update_parameters(self, observations, actions, actions_logpro, rewards, advs, lamb=0.1):
        act_loss = []
        crt_loss = []
        entropy = []

        # 对adv进行归一化
        adv_n = sum(advs, [])
        adv_max = torch.stack(adv_n).max().item()
        adv_min = torch.stack(adv_n).min().item()
        adv_max_min = adv_max-adv_min
        
        # 奖励归一化
        rewards, reward_max, reward_min  = self._compute_adv(actions, rewards)
        rmax_min = reward_max - reward_min
        for action, act_log, reward, adv in zip(actions, actions_logpro, rewards, advs):
            l1 = []
            l2 = []
            e = []
            for a, alp, r, v in zip(action, act_log, reward, adv):
                if a == 1:
                    continue
                # if alp > -1e-5:
                #     alp -= 1e-5
                v = (v-adv_min)/ (adv_max_min)
                r = (r-reward_min)/rmax_min
                # l1.append(-alp * (r-v))
                l1.append(-alp * r)
                e.append(-a * alp)
                l2.append(r - v)

            # actor损失
            act_loss.append(torch.stack(l1).sum())
            # 熵
            entropy.append(torch.stack(e).sum())
            # critic损失
            crt_loss.append(torch.stack(l2).pow(2).sum())

        # 对模型进行熵正则化惩罚
        print()
        act_loss = torch.stack(act_loss).mean()
        entropy = torch.stack(entropy).mean()
        
        # act_loss = act_loss - lamb * entropy.mean()
        self.loss = act_loss.item()
        
        crt_loss = torch.stack(crt_loss).mean()

        # loss = act_loss + crt_loss
        loss = act_loss

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        loss.backward()
        
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # self.critic_optimizer.zero_grad()
        # self.actor_optimizer.zero_grad()


    def pretrain(self, actions, rewards, advs, action_cross_loss):
        # 预训练中，需要对action_logits进行交叉熵损失和value进行均方差损失
        act_loss = []
        crt_loss = []

        # 对adv进行归一化
        adv_std = torch.stack(advs).std().item() + 1e-8
        adv_mean = torch.stack(advs).mean().item()

        # 奖励归一化
        rewards, reward_max, reward_min  = self._compute_adv([actions], [rewards])
        rmax_min = reward_max - reward_min
        
        l1 = []
        l2 = []
        e = []
        for a, r, v in zip(action_cross_loss, rewards[0], advs):
            if a == 0:
                continue
            # if alp > -1e-5:
            #     alp -= 1e-5
            v = (v-adv_mean)/ adv_std
            r = (r-reward_min)/rmax_min
            # l1.append(-alp * (r-v))
            l1.append(a)
            l2.append((r - v).pow(2))

        # actor损失
        act_loss = torch.stack(l1).sum()
        # critic损失
        crt_loss = torch.stack(l2).sum()
        
        self.loss = act_loss.item()

        loss = act_loss + crt_loss
        # loss = act_loss 

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        loss.backward()
        
        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path, weights_only=True):
        self.network.load_state_dict(torch.load(path, weights_only=weights_only))

