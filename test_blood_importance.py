
#%%
import json
from re import A
import pandas as pd
import numpy as np
import os
# Opening JSON file
im_df = pd.read_json('all_norm_Importance.json')
sorted_df_shap = im_df.sort_values(["shap"], ascending=False)
sorted_df_lgbm = im_df.sort_values(["lgbm"], ascending=False)
sorted_df_lime = im_df.sort_values(["lime"], ascending=False)
sorted_df_dice = im_df.sort_values(["dice_global"], ascending=False)
print(sorted_df_shap)
indxs_shap=list(sorted_df_shap.index)
indxs_lgbm=list(sorted_df_lgbm.index)
indxs_lime=list(sorted_df_lime.index)
indxs_d=list(sorted_df_dice.index)
top_6_features_shap=indxs_shap[:5]
top_6_features_lgbm=indxs_lgbm[:5]
top_6_features_lime=indxs_lime[:5]
top_6_features_dice=indxs_d[:5] 

# %%
train_data= pd.read_csv("train_blood.csv")
test_data= pd.read_csv("test_blood.csv")
label_data=pd.read_csv("y_train_blood.csv")
##### shap
shap_data_t=train_data[top_6_features_shap]
shap_data_test=test_data[top_6_features_shap]
shap_data_t.to_csv('shap_data_t.csv', index=False)
shap_data_test.to_csv('shap_data_test.csv', index=False)
################# lgbm
lgbm_data_t=train_data[top_6_features_lgbm]
lgbm_data_test=test_data[top_6_features_lgbm]
lgbm_data_t.to_csv('lgbm_data_t.csv', index=False)
lgbm_data_test.to_csv('lgbm_data_test.csv', index=False)
######## lime
lime_data_t=train_data[top_6_features_lime]
lime_data_test=test_data[top_6_features_lime]
lime_data_t.to_csv('lime_data_t.csv', index=False)
lime_data_test.to_csv('lime_data_test.csv', index=False)
###### dice :

dice_data_t=train_data[top_6_features_dice]
dice_data_test=test_data[top_6_features_dice]
dice_data_t.to_csv('dice_data_t.csv', index=False)
dice_data_test.to_csv('dice_data_test.csv', index=False)
################################################################

def prediction(trian_data,test_data,trian_labels,test_labels,model):
        import pandas as pd
        import numpy as np
        import math
        import json
        import plotly.graph_objs as go
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import plot_roc_curve
        from sklearn.model_selection import cross_val_score
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn import preprocessing
        from collections import Counter
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
        pd.options.display.max_columns = 999
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from sklearn.multiclass import OneVsRestClassifier
        from itertools import cycle
        plt.style.use('ggplot')
        import dice_ml
        from dice_ml.utils import helpers
        from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
        # DiCE imports
        import dice_ml
        from dice_ml.utils import helpers
        #########
        #importing from CSV to Pandas
        trian_data=pd.read_csv("%s" % trian_data);
        test_data=pd.read_csv("%s"% test_data);
        trian_labels=pd.read_csv("%s"% trian_labels);
        test_labels=pd.read_csv("%s"% test_labels);

        col_names=trian_data.columns;
        # loop to change each column to float type
        for col in col_names:
                trian_data[col] = trian_data[col].astype('float',copy=False);
                test_data[col]= test_data[col].astype('float',copy=False);
        #selecting raws with no nan values
        new_train=trian_data
        new_train["Result"]=trian_labels
        new_test=test_data
        new_test["Result"]=test_labels
        trian_data_nonull=new_train.dropna()
        test_data_nonull=new_test.dropna()
        trian_label_nonull=trian_data_nonull["Result"]
        test_label_nonull=test_data_nonull["Result"]
        trian_data_nonull.drop(labels = ["Result",], axis=1,inplace=True )
        test_data_nonull.drop(labels = ["Result",], axis=1,inplace=True )
        for col in col_names:
                trian_data_nonull[col] = trian_data_nonull[col].astype('float',copy=False);
                test_data_nonull[col]= test_data_nonull[col].astype('float',copy=False);
        trian_data_nonull.reset_index(drop=True, inplace=True)
        test_data_nonull.reset_index(drop=True, inplace=True)
        #Scaling using the Standard Scaler
        sc_1=StandardScaler();
        X_1=pd.DataFrame(sc_1.fit_transform(trian_data_nonull));
        X_train, X_val, y_train, y_val = train_test_split(X_1, trian_label_nonull, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0
        test_scale_data=pd.DataFrame(sc_1.fit_transform(test_data_nonull))
        if model =="lgbm_blood":
            #Bulding them Model
            lgbm_clf = lgb.LGBMClassifier(
            max_depth=8,
            num_leaves=2^8,
            min_data_in_leaf=50,
            subsample=0.8,
            random_state=0,
            learning_rate=0.01,
            )

            #Fitting the Model
            lgbm_clf.fit(
                X_train,
                y_train,
                eval_set = [(X_val, y_val)],
                eval_metric="auc",
                )
            preds = lgbm_clf.predict_proba(test_scale_data,num_iteration=100000)[:,1]
            preds_f=lgbm_clf.predict(test_scale_data,num_iteration=100)
            predict_model=lgbm_clf;
        return preds,test_label_nonull,preds_f

# %% Cheack AUC score for classification:

prob_shap,shap_labels_test,preds_shap=prediction("shap_data_t.csv","shap_data_test.csv","train_labels.csv","test_labels.csv","lgbm_blood")
prob_lime,lime_labels_test,preds_lime=prediction("lime_data_t.csv","lime_data_test.csv","train_labels.csv","test_labels.csv","lgbm_blood")
prob_lgbm,lgbm_labels_test,preds_lgbm=prediction("lgbm_data_t.csv","lgbm_data_test.csv","train_labels.csv","test_labels.csv","lgbm_blood")
prob_dice,dice_labels_test,preds_dice=prediction("dice_data_t.csv","dice_data_test.csv","train_labels.csv","test_labels.csv","lgbm_blood")
#%% plots :
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn import metrics
fpr1 , tpr1, thresholds1 = roc_curve(shap_labels_test, prob_shap)
fpr2 , tpr2, thresholds2 = roc_curve(lime_labels_test, prob_lime)
fpr3 , tpr3, thresholds3 = roc_curve(lgbm_labels_test, prob_lgbm)
fpr4 , tpr4, thresholds4 = roc_curve(dice_labels_test, prob_dice)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, label= "Shap",color="red",linewidth=8,alpha=0.3)
#plt.show()
plt.plot(fpr2, tpr2, label= "Lime",color="cyan",linewidth=3)
plt.plot(fpr3, tpr3, label= "LGBM",color="blue" ,alpha=0.3,linewidth=3)
plt.plot(fpr4, tpr4, label= "Dice",color="limegreen",linewidth=3)
plt.legend(loc='best',fontsize = 'xxx-large',prop={'size': 20},facecolor='white',framealpha=0.5,fancybox=True,edgecolor='gray')
plt.xlabel("FPR",color="black",fontsize=20)
plt.ylabel("TPR",color="black",fontsize=20)
plt.title('Receiver Operating Characteristic')
plt.tight_layout()
plt.savefig("Roc_metric_featuers_blood.pdf")
plt.show()
#
from sklearn.metrics import roc_auc_score
print("The AUC Scores for each method : ")
shap_auc_score=roc_auc_score(shap_labels_test, prob_shap)
lime_auc_score=roc_auc_score(lime_labels_test, prob_lime)
lgbm_auc_score=roc_auc_score(lgbm_labels_test, prob_lgbm)
dice_auc=roc_auc_score(dice_labels_test, prob_dice)

print(f'Shap AUC :{shap_auc_score}')
print(f'lime Auc :{lime_auc_score}')
print(f'lgbm auc:{lgbm_auc_score}')
print(f'dice auc:{dice_auc}')

# %%values 
#shap_acc=0.83
#lime_acc=0.78
#lgbm_acc=0.56
#dice_acc=0.77
#acc={"shap_acc":shap_acc,"lime_acc":lime_acc,"lgbm_acc":lgbm_acc,"dice_acc":dice_acc}
#import numpy as np
#import matplotlib.pyplot as plt
#metrics = list(acc.keys())
#values = list(acc.values())
#fig = plt.figure(figsize = (10, 5))
# creating the bar plot
#plt.bar(metrics, values, color ='maroon',
 #       width = 0.4)
 
#plt.xlabel("Importance Metric")
#plt.ylabel("Accuracy")
#plt.title("Model Accuracy by Importance Metric")
#plt.tight_layout()
#plt.show()
# %%
