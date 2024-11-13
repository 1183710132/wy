import numpy as np


def features_extract_func(task, task_instance_features):
    return [task.task_config.cpu, (task.task_config.memory-task_instance_features[3])/(task_instance_features[2]-task_instance_features[3]),
            task.feature['first_layer_task'], task.feature['first_layer_instance'],
            task.feature['layers_task'], task.feature['child_task_numbers']]


def features_normalize_func(x):
    y = (np.array(x) - np.array([0, 0, 0, 0, 0, 0, 1.167, 1.167, 1.5, 1.833, 1.833])) / np.array(
        [2, 1, 1, 1, 100, 1, 0.897, 0.897, 0.957, 1.572, 1.572])
    return y


def features_normalize_func_ac(x):
    y = (np.array(x) - np.array([0, 0, 0, 0, 0, 0, 1.167, 1.167, 1.5, 1.833, 1.833, 0, 0, 0])) / np.array(
        [2, 1, 1, 1, 100, 1, 0.897, 0.897, 0.957, 1.572, 1.572, 1, 1, 1])
    return y
