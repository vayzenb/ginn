import pandas as pd
import os

print(os.listdir())

df = pd.read_csv("Stim/Monkey/monkey_files.csv")

df_deleted = df[df['Delete']==1]

for ii in df_deleted['Monkey']:
    fold = ii[0:3]
    os.remove(f'Stim/Monkey/{fold}/{ii}')

print(df_deleted)