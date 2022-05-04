
#%%
import json
import pandas as pd
import numpy as np
import os
# Opening JSON file
im_df = pd.read_json('all_norm_Importance.json')
sorted_df_shap = im_df.sort_values(["shap"], ascending=False)
print(sorted_df_shap)
indxs=list(sorted_df_shap.index)
top_3_features=indxs[:3]

# %%
train_data= pd.read_csv("train_data.csv")
test_data= pd.read_csv("test_data.csv")
label_data=pd.read_csv("train_labels.csv")
shap_data_t=train_data[top_3_features]
shap_data_test=test_data[top_3_features]
shap_data_t.to_csv('shap_data_t.csv', index=False)
shap_data_test.to_csv('shap_data_test.csv', index=False)
# %%
#check the AUC of the top 5 features
#import os
#command = 'python interpretation.py --train_data "shap_data_t.csv" --test_data "shap_data_test.csv" --train_labels "train_labels.csv" --test_labels "test_labels.csv" --model "lgbm" --feature_imprtance_type "All_normilaze"'
#os.system(command)
# %%values 
shap_acc=0.83
lime_acc=0.78
lgbm_acc=0.56
dice_acc=0.77
acc={"shap_acc":shap_acc,"lime_acc":lime_acc,"lgbm_acc":lgbm_acc,"dice_acc":dice_acc}
import numpy as np
import matplotlib.pyplot as plt
metrics = list(acc.keys())
values = list(acc.values())
fig = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(metrics, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Importance Metric")
plt.ylabel("Accuracy")
plt.title("Model Accuracy by Importance Metric")
plt.tight_layout()
plt.show()
# %%
