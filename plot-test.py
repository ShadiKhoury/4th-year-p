#from cProfile import label
import numpy as np
import pandas as pd
data= pd.read_csv("train_data.csv")
data
label_data=pd.read_csv("train_labels.csv")
corr_matrix = data.corr()
print(corr_matrix["cough"].sort_values(ascending=False))
from matplotlib import pyplot as plt
import seaborn as sns
sns.heatmap(corr_matrix,vmax=1,square=True)

fig = plt.figure(figsize = (20, 25))
j = 0
for i in data.columns:
    plt.subplot(2, 4, j+1)
    j += 1
    sns.distplot(data[i][label_data['corona_result']==0], color='r', label = 'Negative')
    sns.distplot(data[i][label_data['corona_result']==1], color='g', label = 'Positive')
    plt.legend(loc='best')
fig.suptitle('Covid Data Analysis')
fig.subplots_adjust(top=0.95), fig.subplots_adjust(bottom=0.05),fig.subplots_adjust(hspace=0.5), fig.subplots_adjust(wspace=0.5)
fig.tight_layout()

plt.show()