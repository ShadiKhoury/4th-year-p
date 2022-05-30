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
#Correlation with output variable
from sklearn.feature_selection import mutual_info_regression
discrete_features = data.dtypes == int

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(data, label_data['corona_result'], discrete_features)
mi_scores[::3]

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

# bloodStream

data= pd.read_csv("train_blood.csv")
data
label_data=pd.read_csv("y_train_blood.csv")
corr_matrix = data.corr()
#print(corr_matrix["cough"].sort_values(ascending=False))
from matplotlib import pyplot as plt
import seaborn as sns
sns.heatmap(corr_matrix,vmax=1,square=True)

fig = plt.figure(figsize = (20, 25))
j = 0
for i in data.columns:
    plt.subplot(5, 4, j+1)
    j += 1
    sns.distplot(data[i][label_data['Death']==0], color='r')
    sns.distplot(data[i][label_data['Death']==1], color='g')
    plt.legend(loc='best')
fig.suptitle('Blood Data Analysis')
fig.subplots_adjust(top=0.95), fig.subplots_adjust(bottom=0.05),fig.subplots_adjust(hspace=0.5), fig.subplots_adjust(wspace=0.5)
fig.tight_layout()

plt.show()

corr_blood = data.corr(label_data['Death'])
sns.heatmap(corr_blood, annot=True, cmap=plt.cm.Reds)
plt.show()