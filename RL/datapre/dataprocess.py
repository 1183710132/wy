import xmltodict
import pandas as pd
import os

directory = 'RL\\xml'
for fileName in os.listdir(directory):
    if not fileName.endswith('.xml'):
        continue
    # 读取 XML 文件
    with open(directory+'\\'+fileName, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # 将 XML 转换为字典
    data_dict = xmltodict.parse(xml_content)
    data_dict = data_dict['adag']
    task_dict = data_dict['job']
    task_edge = data_dict['child']

    columns = ['task_id', 'parents', 'job_id', 'instances_num', 'task_type', 'status', 'start_time', 'end_time', 'cpu', 'memory', 'duration', 'submit_time', 'disk']

    dataFrame = {}
    for col in columns:
        dataFrame[col] = []

    parents = {}
    for i in task_edge:
        parents[i['@ref']] = []
        if len(i['parent']) > 1:
            parents[i['@ref']] = [ p['@ref'] for p in i['parent']]
        else:
            parents[i['@ref']].append(i['parent']['@ref'])

    for i in task_dict:
        # task_id = 'task_' + str(int(i['@id'][2:]))
        task_id = i['@id']
        dataFrame['task_id'].append(task_id)
        dataFrame['task_type'].append(i['@namespace'])
        dataFrame['duration'].append(min(float(i['@runtime']), 10.0))
        # dataFrame['instances_num'].append(1)
        dataFrame['instances_num'].append(min(len(i['uses']), 10))
        memory = sum([int(u['@size']) for u in i['uses']])
        dataFrame['memory'].append(memory/1024)
        dataFrame['status'].append('Terminated')
        dataFrame['cpu'].append(1)
        dataFrame['job_id'].append(1)
        dataFrame['start_time'].append('0')
        dataFrame['end_time'].append('100000000000')
        dataFrame['submit_time'].append('0')
        dataFrame['disk'].append(0)
        if task_id in parents.keys():
            dataFrame['parents'].append(parents[task_id])
        else:
            dataFrame['parents'].append([])

    # 打印转换后的字典
    # print(data_dict)
    df = pd.DataFrame(dataFrame)
    fileName = fileName.replace('xml', 'csv')
    df.to_csv(directory.replace('xml', 'csv') + '\\' + fileName)
