import xmltodict
import pandas as pd
import os
import ast

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
    job_dict = data_dict['job']
    job_edge = data_dict['child']

    columns = ['task_id', 'parents', 'job_id', 'instances_num', 'task_type', 'status', 'start_time', 'end_time', 'cpu', 'memory', 'duration', 'submit_time', 'disk']

    dataFrame = {}
    for col in columns:
        dataFrame[col] = []

    parents = {}
    for i in job_edge:
        parents[i['@ref']] = []
        if len(i['parent']) > 1:
            parents[i['@ref']] = [ p['@ref'] for p in i['parent']]
        else:
            parents[i['@ref']].append(i['parent']['@ref'])

    for job in job_dict:
        job_id = job['@id']
        for task, task_id in zip(job['uses'], range(len(job['uses']))):
            dataFrame['task_id'].append(job_id + '_' +str(task_id))
            dataFrame['task_type'].append(job['@namespace'])
            dataFrame['duration'].append(1)
            # dataFrame['instances_num'].append(1)
            dataFrame['instances_num'].append(1)
            memory = int(task['@size'])
            dataFrame['memory'].append(memory/1024)
            dataFrame['status'].append('Terminated')
            dataFrame['cpu'].append(1)
            dataFrame['job_id'].append(job_id)
            dataFrame['start_time'].append('0')
            dataFrame['end_time'].append('100000000000')
            dataFrame['submit_time'].append('0')
            dataFrame['disk'].append(0)
            if job_id in parents.keys():
                dataFrame['parents'].append(str(parents[job_id]))
            else:
                dataFrame['parents'].append(str([]))

    df = pd.DataFrame(dataFrame)
    # for i in range(len(df)):
    #     series = df.iloc[i]
    #     parents_task = []
    #     for parent in ast.literal_eval(series.parents):
    #         find_df = df[df['job_id'] == parent]
    #         parents_task += [d for d in find_df['task_id']]
    #     df.loc[i, 'parents'] = str(parents_task)

    fileName = fileName.replace('xml', 'csv')
    df.to_csv('RL\\csv_2\\' + fileName)