import pandas as pd
import os

file_name = os.getcwd()+'/jobs_files/job.csv'

df = pd.read_csv(file_name)

parents = []
for i in range(len(df)):
    series = df.iloc[i]
    task_id = series.task_id
    tmp = task_id.split('_')
    parents.append(str(tmp[1:]))
    df.loc[i, 'task_id'] = tmp[0]
    df.loc[i, 'job_id'] = tmp[0]

df['parents'] = parents
df.to_csv(os.getcwd() + '/jobs_files/job_1.csv')