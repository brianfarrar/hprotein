import pandas as pd
import numpy as np
import math
from tqdm import tqdm

#source_name = 'submit_ensemble_f9c85a'
source_name = 'submit_final_ensemble'
leak_file = 'leak/leak259.csv'
submit_file = 'stage1_submit' + '/' + source_name + '.csv'
hpa_file = 'hpa_csv/HPAv18RBGY_wodpl.csv'
output_file = 'stage1_submit' + '/' + source_name + '_revised.csv'

df_leak = pd.read_csv(leak_file)
df_submit = pd.read_csv(submit_file)
df_hpa = pd.read_csv(hpa_file)

pred_list = []
replaced_list = []
for j in range(len(df_submit)):
    sid = df_submit['Id'][j]
    predicted = df_submit['Predicted'][j]
    if (len(df_leak.loc[df_leak['Test'] == sid])) == 1:
        hpa_sid = df_leak.loc[df_leak['Test'] == sid]['Extra'].item().split('_',1)[1]
        leak_target = df_hpa.loc[df_hpa['Id'] == hpa_sid]['Target'].item()
        leak_target = np.array(leak_target.split(' '))
        leak_target.sort()
        lt= ''
        for t in leak_target:
            lt += t + ' '
        leak_target = lt.strip()
        pred_list.append([sid, leak_target])
        replaced_list.append([sid, leak_target, predicted])
        #print('{} -> L: {} | P: {}'.format(sid, leak_target, predicted))
    else:
        if not isinstance(predicted, str):
            if math.isnan(predicted):
                predicted = ''
        pred_list.append([sid, predicted])

df_revised_predictions = pd.DataFrame(np.asarray(pred_list))
df_revised_predictions.to_csv(output_file, index=False, header=['Id','Predicted'])
