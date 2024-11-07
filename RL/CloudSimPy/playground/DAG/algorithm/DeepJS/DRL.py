import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class Node(object):
    def __init__(self, observation, action, reward, clock):
        # 定义观测状态，动作空间，奖励和时钟
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class RLAlgorithm(object):
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []

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
        if len(all_candidates) == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
            return None, None
        else:
            # 提取候选集的特征，转成tensor
            features = self.extract_features(all_candidates)
            features = tf.convert_to_tensor(features, dtype=np.float32)
            # agent训练，给出logits，即每个动作的概率
            logits = self.agent.brain(features)
            # pair_index = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=1).numpy()[0]
            # 采样一个动作选择，动作即是将task i调度到机器j中，其中采样概率是根据logits中的调度概率来进行采样的而不是均匀随机采样
            pair_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1).numpy()[0]
            # 创建当前调度的节点
            node = Node(features, pair_index, 0, clock)
            target_machine = all_candidates[pair_index][0]
            target_task = all_candidates[pair_index][1]
            print('machine:{}, task:{}, clock:{}'.format(target_machine.id, target_task.task_index, clock))
            self.current_trajectory.append(node)

        return all_candidates[pair_index]
