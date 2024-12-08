import ast

# 父节点信息提取
def father_task_indices(task_id, task_type):
    father_indices = []

    if task_id.find('task_') != -1:
        task_index = task_type + '_' + 'task_id'
        return task_index, father_indices

    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    start_index = -1

    for i, char_s in enumerate(task_id):
        if (char_s in num_list) and (start_index == -1):
            start_index = i
        if (char_s not in num_list) and (start_index != -1):
            father_index = task_type + '_' + task_id[start_index: i]
            father_indices.append(father_index)
            start_index = -1

    if start_index != -1:
        father_index = task_type + '_' + task_id[start_index:]
        father_indices.append(father_index)

    task_index = father_indices[0]
    father_indices = father_indices[1:]

    return task_index, father_indices


# 子节点特征提取
def task_features(job):
    child_indices = {}
    father_indices = {}
    tasks = job.tasks_map.values()
    task_features = {}
    # 有多少孩子节点
    # 有多少同级节点
    # 同级节点memory占比
    layer_task_num = len(tasks)
    memory_count = 0
    for task in tasks:
        task_index = task.task_index
        task_feature = {}
        task_feature['children_task_num'] = len(task.task_config.children)
        task_feature['layer_task_num'] = layer_task_num
        task_feature['task_memory_percent'] = task.task_config.memory
        task_feature['parents_job_num'] = len(task.task_config.parent_indices)
        memory_count += task.task_config.memory
        task_features[task_index] = task_feature
    for task in tasks:
        task_index = task.task_index
        task_features[task_index]['task_memory_percent'] = task.task_config.memory / memory_count
    return task_features


# 权值计算
def weights_calculate(tasks):
    weight_tasks = {}
    for task in tasks:
        feature = task.feature
        weight = feature['first_layer_task'] + feature['first_layer_instance'] + feature['layers_task'] + feature[
            'child_task_numbers'] + feature['child_instance_numbers']
        task_list = weight_tasks.setdefault(weight, [])
        task_list.append(task)

    sorted_weights = sorted(weight_tasks.keys(), reverse=True)
    sorted_tasks = []
    for weight in sorted_weights:
        sorted_tasks.extend(weight_tasks[weight])

    return sorted_tasks
