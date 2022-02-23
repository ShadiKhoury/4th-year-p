from matplotlib.pyplot import figure


def interpretation (trian_data,test_data,trian_labels,test_labels,model,feature_imprtance_type):
    #imports 
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
    
    #Scaling using the Standard Scaler
    sc_1=StandardScaler();
    X_1=pd.DataFrame(sc_1.fit_transform(trian_data));
    X_train, X_val, y_train, y_val = train_test_split(X_1, trian_labels, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0
    test_scale_data=pd.DataFrame(sc_1.fit_transform(test_data))
    if model =="lgbm": 

        #Bulding them Model
        lgbm_clf = lgb.LGBMClassifier(
        num_leaves= 20,
        min_data_in_leaf= 4,
        feature_fraction= 0.2,
        bagging_fraction=0.8,
        bagging_freq=5,
        learning_rate= 0.05,
        verbose=1,
        num_boost_round=603,
        early_stopping_rounds=5,
        metric="auc",
        objective = 'binary',)

        #Fitting the Model
        lgbm_clf.fit(
            X_train,
            y_train,
            eval_set = [(X_val, y_val)],
            eval_metric="auc",
            )
        preds = lgbm_clf.predict_proba(test_scale_data,num_iteration=100)
        predict_model=lgbm_clf;
    #else:

    #from sklearn import metrics
    #metrics.plot_roc_curve(model, test_scale_data, test_labels) 


    ## Feature importance type

    if feature_imprtance_type=="SHAP":
        import shap
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')
        # load JS visualization code to notebook
        shap.initjs()
        explainer = shap.TreeExplainer(predict_model)
        shap_values = explainer.shap_values(test_scale_data);
        #shap.summary_plot(shap_values[1], features=test_scale_data, feature_names=test_data.columns,)
        shap_sum = np.abs(shap_values[1]).mean(axis=0)
        importance_df_shap = pd.DataFrame([test_data.columns.tolist(), shap_sum.tolist()]).T
        importance_df_shap.columns = ['name', 'importance']
        importance_df_shap = importance_df_shap.sort_values('importance', ascending=False)
        importance_df_shap.reset_index(drop=True)
        ##PLOTING THE SHAP IMPORTANCE 
        figure();
        figure(figsize=(10, 10), dpi=80);
        importance_df_shap.plot.barh(y="importance",x="name",);
        plt.gca().invert_yaxis()
        plt.savefig("shap_importance.pdf")
        #plt.show()
        #importance_dict_shap=importance_df_shap.to_dict(orient='index');
        importance_dict_shap = importance_df_shap.set_index('name').T.to_dict('records')[0]
        max_shap=max(importance_df_shap.importance);
        normal_shap=[]
        for i in importance_df_shap.importance:
            normlize=i/max_shap;
            normal_shap.append(normlize);
        importance_df_shap_norm = pd.DataFrame([test_data.columns.tolist(), normal_shap]).T
        importance_df_shap_norm.columns = ['name', 'importance normlized']
        importance_df_shap_norm = importance_df_shap_norm.sort_values('importance normlized', ascending=False)
        importance_df_shap_norm.reset_index(drop=True)
        #figure();
        #figure(figsize=(10, 10), dpi=80);
        #importance_df_shap_norm.plot.barh(y="importance normlized",x="name",);
        #plt.gca().invert_yaxis()
        #plt.savefig("shap_importance_normlized.pdf")
        ########### For ploting
        #js_shap = importance_dict_shap.to_json(orient = "values");
        #parsed = json.loads(js_shap);
        #print(json.dumps(parsed,))
        #EXPORTING THE SHAP JSON FILE.
        print(json.dumps(importance_dict_shap,))
        with open('SHAP_Feature_Importance.json', 'w') as outfile:
            return json.dump(importance_dict_shap,outfile)




    #lime imprtance
    elif feature_imprtance_type=="lime":
        from lime.lime_tabular import LimeTabularExplainer
        import lime
        X_train = X_train.to_numpy()
        X_val= X_val.to_numpy()
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        num_f=len(test_data.columns)
        explainer = LimeTabularExplainer(X_train,mode="classification", 
                                 feature_names=test_data.columns, 
                                 class_names=["Postive","Negative",],
                                 discretize_continuous=False)
        i = np.random.randint(0, X_val.shape[0]) 
        exp = explainer.explain_instance(X_val[37157],  lgbm_clf.predict_proba, num_features=num_f,)#need to change to i
        a=exp.as_list()
        importance_df_Lime= pd.DataFrame(a)
        importance_df_Lime.columns = ['name', 'importance']
        importance_df_Lime = importance_df_Lime.sort_values('importance', ascending=False)
        #Ploting
        importance_df_Lime.plot.barh(y="importance",x="name",color="#66FF66");
        plt.gca().invert_yaxis()
        plt.savefig("lime_importance.pdf")
        max_lime=max(importance_df_Lime.importance);
        normal_lime=[]
        for i in importance_df_Lime.importance:
            normlize=i/max_lime;
            normal_lime.append(normlize);
        importance_df_lime_norm = pd.DataFrame([test_data.columns.tolist(), normal_lime]).T
        importance_df_lime_norm.columns = ['name', 'importance normlized']
        importance_df_lime_norm = importance_df_lime_norm.sort_values('importance normlized', ascending=False)
        importance_df_lime_norm.reset_index(drop=True)
        #plt.show()
        #js_Lime = importance_df_Lime.to_json(orient = "values")
        #parsed_l = json.loads(js_Lime)
        #Json
        importance_dict_lime = importance_df_Lime.set_index('name').T.to_dict('records')[0]
        with open('Lime_Feature_Importance.json', 'w') as outfile:
            return json.dump(importance_dict_lime,outfile)





    #lgbm_plot_imprtance
    elif feature_imprtance_type=="lgbm_plot_importance":
        feature_imp = pd.DataFrame(sorted(zip(lgbm_clf.feature_importances_,trian_data.columns)), columns=['Value','Feature'])
        importance_df_lgbm=feature_imp
        importance_df_lgbm.columns = ['name', 'importance']
        importance_df_lgbm=importance_df_lgbm[['importance','name']]
        importance_df_lgbm.columns = ['name', 'importance']
        importance_df_lgbm = importance_df_lgbm.sort_values('importance', ascending=False)
        #Ploting
        importance_df_lgbm.plot.barh(y="importance",x="name",color="#00008B");
        plt.gca().invert_yaxis()
        plt.savefig("lgbm_importance.pdf")
        max_lgbm=max(importance_df_lgbm.importance);
        normal_lgbm=[]
        for i in importance_df_lgbm.importance:
            normlize=i/max_lgbm;
            normal_lgbm.append(normlize);
        importance_df_lgbm_norm = pd.DataFrame([test_data.columns.tolist(), normal_lgbm]).T
        importance_df_lgbm_norm.columns = ['name', 'importance normlized']
        importance_df_lgbm_norm = importance_df_lgbm_norm.sort_values('importance normlized', ascending=False)
        importance_df_lgbm_norm.reset_index(drop=True)
        #plt.show()
        #js_lgbm = importance_df_lgbm.to_json(orient = "values")
        #parsed_2 = json.loads(js_lgbm)
        #Json
        importance_dict_lgbm = importance_df_lgbm.set_index('name').T.to_dict('records')[0]
        with open('Lgbm_Feature_Importance.json', 'w') as outfile:
            return json.dump(importance_dict_lgbm,outfile)




    #dice local with cf
    elif feature_imprtance_type=="dice_local_cf":
        trainn_data=trian_data;
        trainn_data["labels"]=trian_labels
        dicedata = dice_ml.Data(dataframe=trainn_data,continuous_features=[], outcome_name="labels")
        # Using sklearn backend
        m = dice_ml.Model(model=predict_model, backend="sklearn",model_type = 'classifier')
        # Using method=random for generating CFs
        exp_dice = dice_ml.Dice(dicedata, m, method="random")
        query_instance=test_data[4:5];
        e1 = exp_dice.generate_counterfactuals(query_instance, total_CFs=10, 
                                       desired_class="opposite",
                                       verbose=False,
                                       features_to_vary="all")
        #Local Feature Importance Scores with Counterfactuals list
        imp = exp_dice.local_feature_importance(query_instance, cf_examples_list=e1.cf_examples_list);
        result = imp.local_importance[0].items()
        # Convert object to a list
        data_imp = list(result);
        feature_imp1 = pd.DataFrame(sorted(data_imp), columns=['Feature','Value'])
        importance_df_dice_local_cf=feature_imp1
        importance_df_dice_local_cf.columns = ['name', 'importance'];
        importance_df_dice_local_cf = importance_df_dice_local_cf.sort_values('importance', ascending=False)
        #Ploting
        importance_df_dice_local_cf.plot.barh(y="importance",x="name",color="#FF6103");
        plt.gca().invert_yaxis()
        plt.savefig("dice_local_cf_importance.pdf")
        #plt.show()
        #js_dice_lc_cf = importance_df_dice_local_cf.to_json(orient = "values")
        #parsed_3 = json.loads(js_dice_lc_cf)

        #Json
        importance_dict_dicecflo = importance_df_dice_local_cf.set_index('name').T.to_dict('records')[0]
        with open('Dice_local_cf_Feature_Importance.json', 'w') as outfile:
            return json.dump(importance_dict_dicecflo,outfile)



    #Dice local
    elif feature_imprtance_type=="dice_local":
        trainn_data=trian_data;
        trainn_data["labels"]=trian_labels
        dicedata = dice_ml.Data(dataframe=trainn_data,continuous_features=[], outcome_name="labels")
        # Using sklearn backend
        m = dice_ml.Model(model=predict_model, backend="sklearn",model_type = 'classifier')
        # Using method=random for generating CFs
        exp_dice = dice_ml.Dice(dicedata, m, method="random")
        query_instance=test_data[4:5];
        imp = exp_dice.local_feature_importance(query_instance, posthoc_sparsity_param=None);
        result = imp.local_importance[0].items()
        # Convert object to a list
        data_imp = list(result);
        feature_imp1 = pd.DataFrame(sorted(data_imp), columns=['Feature','Value'])
        importance_df_dice_local=feature_imp1
        importance_df_dice_local.columns = ['name', 'importance'];
        importance_df_dice_local = importance_df_dice_local.sort_values('importance', ascending=False)
        #Ploting
        importance_df_dice_local.plot.barh(y="importance",x="name",color="#FFB90F");
        plt.gca().invert_yaxis()
        plt.savefig("dice_local_importance.pdf")
        #plt.show()
        #js_dice_lc_cf = importance_df_dice_local_cf.to_json(orient = "values")
        #parsed_4 = json.loads(js_dice_lc_cf)
        #Json
        importance_dict_dice_lo = importance_df_dice_local.set_index('name').T.to_dict('records')[0]
        with open('Dice_local__Feature_Importance.json', 'w') as outfile:
            return json.dump(importance_dict_dice_lo,outfile)



    elif feature_imprtance_type=="dice_global":
        trainn_data=trian_data;
        trainn_data["labels"]=trian_labels
        dicedata = dice_ml.Data(dataframe=trainn_data,continuous_features=[], outcome_name="labels")
        # Using sklearn backend
        m = dice_ml.Model(model=predict_model, backend="sklearn",model_type = 'classifier')
        # Using method=random for generating CFs
        exp_dice = dice_ml.Dice(dicedata, m, method="random")
        query_instance=test_data[0:10];
        cobj = exp_dice.global_feature_importance(query_instance, total_CFs=10, posthoc_sparsity_param=None)
        result = cobj.summary_importance.items()
        # Convert object to a list
        data_imp = list(result)
        feature_imp1 = pd.DataFrame(sorted(data_imp), columns=['Feature','Value'])
        importance_df_dice_global=feature_imp1
        importance_df_dice_global.columns = ['name', 'importance'];
        importance_df_dice_global = importance_df_dice_global.sort_values('importance', ascending=False)
        #Ploting
        importance_df_dice_global.plot.barh(y="importance",x="name",color="#BF3EFF");
        plt.gca().invert_yaxis()
        plt.savefig("dice_global.pdf")
        #plt.show()
        #js_dice_global = importance_df_dice_global.to_json(orient = "values")
        #parsed_5 = json.loads(js_dice_global)
        #Json
        importance_dict_dice_glo = importance_df_dice_global.set_index('name').T.to_dict('records')[0]
        with open('Dice_global_Feature_Importance.json', 'w') as outfile:
            return json.dump(importance_dict_dice_glo,outfile)

    elif feature_imprtance_type=="All_normilaze":


        ## Shap ##
        import shap
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')
        # load JS visualization code to notebook
        shap.initjs()
        explainer = shap.TreeExplainer(predict_model)
        shap_values = explainer.shap_values(test_scale_data);
        #shap.summary_plot(shap_values[1], features=test_scale_data, feature_names=test_data.columns,)
        shap_sum = np.abs(shap_values[1]).mean(axis=0)
        importance_df_shap = pd.DataFrame([test_data.columns.tolist(), shap_sum.tolist()]).T
        importance_df_shap.columns = ['name', 'importance']
        importance_df_shap = importance_df_shap.sort_values('importance', ascending=False)
        importance_df_shap.reset_index(drop=True)
        
    
        max_shap=max(importance_df_shap.importance);
        normal_shap=[]
        for i in importance_df_shap.importance:
            normlize=i/max_shap;
            normal_shap.append(normlize);
        importance_df_shap_norm = pd.DataFrame([test_data.columns.tolist(), normal_shap]).T
        importance_df_shap_norm.columns = ['name', 'importance_normlized']
        importance_df_shap_norm = importance_df_shap_norm.sort_values('importance_normlized', ascending=False)
        importance_df_shap_norm.reset_index(drop=True)


        ## lime ##
        from lime.lime_tabular import LimeTabularExplainer
        import lime
        X_train = X_train.to_numpy()
        X_val= X_val.to_numpy()
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        num_f=len(test_data.columns)
        explainer = LimeTabularExplainer(X_train,mode="classification", 
                                 feature_names=test_data.columns, 
                                 class_names=["Postive","Negative",],
                                 discretize_continuous=False)
        i = np.random.randint(0, X_val.shape[0]) 
        exp = explainer.explain_instance(X_val[37157],  lgbm_clf.predict_proba, num_features=num_f,)#need to change to i
        a=exp.as_list()
        importance_df_Lime= pd.DataFrame(a)
        importance_df_Lime.columns = ['name', 'importance']
        importance_df_Lime = importance_df_Lime.sort_values('importance', ascending=False)
        
        max_lime=max(importance_df_Lime.importance);
        normal_lime=[]
        for i in importance_df_Lime.importance:
            normlize=i/max_lime;
            normal_lime.append(normlize);
        importance_df_lime_norm = pd.DataFrame([test_data.columns.tolist(), normal_lime]).T
        importance_df_lime_norm.columns = ['name', 'importance_normlized']
        importance_df_lime_norm = importance_df_lime_norm.sort_values('importance_normlized', ascending=False)
        importance_df_lime_norm.reset_index(drop=True)

        ##lgbm##

        feature_imp = pd.DataFrame(sorted(zip(lgbm_clf.feature_importances_,trian_data.columns)), columns=['Value','Feature'])
        importance_df_lgbm=feature_imp
        importance_df_lgbm.columns = ['name', 'importance']
        importance_df_lgbm=importance_df_lgbm[['importance','name']]
        importance_df_lgbm.columns = ['name', 'importance']
        importance_df_lgbm = importance_df_lgbm.sort_values('importance', ascending=False)
        
        max_lgbm=max(importance_df_lgbm.importance);
        normal_lgbm=[]
        for i in importance_df_lgbm.importance:
            normlize=i/max_lgbm;
            normal_lgbm.append(normlize);
        importance_df_lgbm_norm = pd.DataFrame([test_data.columns.tolist(), normal_lgbm]).T
        importance_df_lgbm_norm.columns = ['name', 'importance_normlized']
        importance_df_lgbm_norm = importance_df_lgbm_norm.sort_values('importance_normlized', ascending=False)
        importance_df_lgbm_norm.reset_index(drop=True)

        ## Dice_local_cf ##

        trainn_data=trian_data;
        trainn_data["labels"]=trian_labels
        dicedata = dice_ml.Data(dataframe=trainn_data,continuous_features=[], outcome_name="labels")
        # Using sklearn backend
        m = dice_ml.Model(model=predict_model, backend="sklearn",model_type = 'classifier')
        # Using method=random for generating CFs
        exp_dice = dice_ml.Dice(dicedata, m, method="random")
        query_instance=test_data[4:5];
        e1 = exp_dice.generate_counterfactuals(query_instance, total_CFs=10, 
                                       desired_class="opposite",
                                       verbose=False,
                                       features_to_vary="all")
        #Local Feature Importance Scores with Counterfactuals list
        imp = exp_dice.local_feature_importance(query_instance, cf_examples_list=e1.cf_examples_list);
        result = imp.local_importance[0].items()
        # Convert object to a list
        data_imp = list(result);
        feature_imp1 = pd.DataFrame(sorted(data_imp), columns=['Feature','Value'])
        importance_df_dice_local_cf=feature_imp1
        importance_df_dice_local_cf.columns = ['name', 'importance'];
        importance_df_dice_local_cf = importance_df_dice_local_cf.sort_values('importance', ascending=False)
        max_dice_local_cf=max(importance_df_dice_local_cf.importance);
        normal_dice_loc_cf=[]
        for i in importance_df_dice_local_cf.importance:
            normlize=i/max_dice_local_cf;
            normal_dice_loc_cf.append(normlize);
        importance_df_dice_local_cf_norm = pd.DataFrame([test_data.columns.tolist(), normal_dice_loc_cf]).T
        importance_df_dice_local_cf_norm.columns = ['name', 'importance_normlized']
        importance_df_dice_local_cf_norm = importance_df_dice_local_cf_norm.sort_values('importance_normlized', ascending=False)
        importance_df_dice_local_cf_norm.reset_index(drop=True)

        ## Dice local ##
        trainn_data=trian_data;
        trainn_data["labels"]=trian_labels
        dicedata = dice_ml.Data(dataframe=trainn_data,continuous_features=[], outcome_name="labels")
        # Using sklearn backend
        m = dice_ml.Model(model=predict_model, backend="sklearn",model_type = 'classifier')
        # Using method=random for generating CFs
        exp_dice = dice_ml.Dice(dicedata, m, method="random")
        query_instance=test_data[4:5];
        imp = exp_dice.local_feature_importance(query_instance, posthoc_sparsity_param=None);
        result = imp.local_importance[0].items()
        # Convert object to a list
        data_imp = list(result);
        feature_imp1 = pd.DataFrame(sorted(data_imp), columns=['Feature','Value'])
        importance_df_dice_local=feature_imp1
        importance_df_dice_local.columns = ['name', 'importance'];
        importance_df_dice_local = importance_df_dice_local.sort_values('importance', ascending=False)
        max_dice_local=max(importance_df_dice_local.importance);
        normal_dice_loc=[]
        for i in importance_df_dice_local.importance:
            normlize=i/max_dice_local;
            normal_dice_loc.append(normlize);
        importance_df_dice_local_norm = pd.DataFrame([test_data.columns.tolist(), normal_dice_loc]).T
        importance_df_dice_local_norm.columns = ['name', 'importance_normlized']
        importance_df_dice_local_norm = importance_df_dice_local_cf_norm.sort_values('importance_normlized', ascending=False)
        importance_df_dice_local_norm.reset_index(drop=True)

        ## Dice global ##
        trainn_data=trian_data;
        trainn_data["labels"]=trian_labels
        dicedata = dice_ml.Data(dataframe=trainn_data,continuous_features=[], outcome_name="labels")
        # Using sklearn backend
        m = dice_ml.Model(model=predict_model, backend="sklearn",model_type = 'classifier')
        # Using method=random for generating CFs
        exp_dice = dice_ml.Dice(dicedata, m, method="random")
        query_instance=test_data[0:10];
        cobj = exp_dice.global_feature_importance(query_instance, total_CFs=10, posthoc_sparsity_param=None)
        result = cobj.summary_importance.items()
        # Convert object to a list
        data_imp = list(result)
        feature_imp1 = pd.DataFrame(sorted(data_imp), columns=['Feature','Value'])
        importance_df_dice_global=feature_imp1
        importance_df_dice_global.columns = ['name', 'importance'];
        importance_df_dice_global = importance_df_dice_global.sort_values('importance', ascending=False)

        max_dice_global=max(importance_df_dice_global.importance);
        normal_dice_g=[]
        for i in importance_df_dice_global.importance:
            normlize=i/max_dice_global;
            normal_dice_g.append(normlize);
        importance_df_dice_g_norm = pd.DataFrame([test_data.columns.tolist(), normal_dice_g]).T
        importance_df_dice_g_norm.columns = ['name', 'importance_normlized']
        importance_df_dice_g_norm = importance_df_dice_local_cf_norm.sort_values('importance_normlized', ascending=False)
        importance_df_dice_g_norm.reset_index(drop=True)



        all_normlize_pd=pd.DataFrame({'shap_norm':importance_df_shap_norm.importance_normlized.values,'lime_norm':importance_df_lime_norm.importance_normlized.values,
        'lgbm_norm':importance_df_lgbm_norm.importance_normlized.values,'dice_local_cf_norm':importance_df_dice_local_cf_norm.importance_normlized.values,
        'dice_loc_norm':importance_df_dice_local_norm.importance_normlized.values,'dice_glo_norm':importance_df_dice_g_norm.importance_normlized.values},index=test_data.columns.tolist())
        #Ploting all 
        
        
        all_normlize_pd.plot.barh();
        plt.gca().invert_yaxis()
        plt.savefig("normalize.pdf")
        plt.show()
        

        importance_all_norm = all_normlize_pd.to_dict('index')
        with open('all_norm_Importance.json', 'w') as outfile:
            return json.dump(importance_all_norm,outfile)










        

    





    



        


    


   
          

