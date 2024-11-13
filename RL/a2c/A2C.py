import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Actor_Critic(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor_Critic, self).__init__()
        self.actor_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.critic_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

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

class Agent(object):

    def __init__(self, hidden_dim, input_dim, lr=0.0001, gamma=0.98):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.gamma = gamma
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.network = Actor_Critic(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.network.actor_layer.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.network.critic_layer.parameters(), lr=lr)

    def select_action(self, observation, temperature=1):
        logits, values = self.network.forward(observation, t=temperature)
        # 这里有个bug，如果只有一个候选时，直接squeeze shape=[]，应该保留一维
        return torch.squeeze(logits, dim=-1), values
    

    def _compute_adv(self, actions, rewards):
        res = []
        rt = 0
        for a, r in zip(reversed(actions), reversed(rewards)):
            # if a is None:
            #     rt = rt + r
            # else:
            rt = rt * self.gamma + r
            res.append(rt)
        return res

    def update_parameters(self, observations, actions, actions_logpro, rewards, advs, lamb=1):
        act_loss = []
        crt_loss = []
        entropy = []

        # 对adv进行归一化        
        adv_n = sum(advs, [])
        adv_mean = torch.stack(adv_n).mean()
        adv_std = torch.stack(adv_n).std()

        for action, act_log, reward, adv in zip(actions, actions_logpro, rewards, advs):
            l1 = []
            l2 = []
            e = []
            reward = self._compute_adv(actions, reward)
            for a, alp, r, v in zip(action, act_log, reward, adv):
                if a is None:
                    continue
                v = (v-adv_mean)/ (adv_std + 0.0001)
                l1.append(-alp * (r-v))
                e.append(-a * alp)
                l2.append(r - v)

            # actor损失
            act_loss.append(torch.stack(l1))
            # 熵
            entropy.append(torch.stack(e))
            # critic损失
            crt_loss.append(torch.stack(l2))

        # 对模型进行熵正则化惩罚
        act_loss = torch.cat(act_loss, dim=0).mean(dim=0)
        entropy = torch.cat(entropy, dim=0).sum(dim=0)
        
        act_loss = act_loss.mean()
        self.loss = act_loss.item()
        
        crt_loss = torch.cat(crt_loss, dim=0).pow(2).mean()

        loss = act_loss + crt_loss
        loss.backward()

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path, weights_only=True):
        self.network.load_state_dict(torch.load(path, weights_only=weights_only))
