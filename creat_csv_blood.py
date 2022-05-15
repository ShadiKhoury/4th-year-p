import numpy as np
import pandas as pd

cols_to_filter = ["kupa_", "diag_", "UnitID_", "Client_", "birth_country_",
                      "pat_nation_"] #category_candida
    #13 december
    # cols_to_filter += ["Basophils_No_", "Basophils_", "Eosinophils__", "Eosinophils_No_", "Lymphocytes__", "Lymphocytes_No", "Mono_", "Monocytes_No_", "NRBC_100_WBC"]
    
correlated_features = {'Diabetes_Drugs_Flg': 'Insulin_Drugs_Flg',
                           'category_gram_negative': 'category_gram_positive',
                           'WBC': ('Basophils_No_', 'Basophils_', 'Eosinophils__', 'Eosinophils_No_',
                                             'Lymphocytes_No','Lymphocytes__', 'Mono_', 'Monocytes_No_',
                                              'Neutrophils__', 'Neutrophils_No_'),
                           }
train_path = r'C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\BloodStream\data\data\real_train.csv'
test_path = r'C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\BloodStream\data\data\real_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

cols = [c for c in train.columns if not any(s in c for s in cols_to_filter)]


y_train = train["Death"]
y_test = test["Death"]
with open(r"C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\BloodStream\feature_lists\feature_lists\20_features.txt") as file:
    cols = [line.strip() for line in file]

x_train = train[cols].copy().reset_index(drop=True)
x_test = test[cols].copy().reset_index(drop=True)

x_train.to_csv("train_blood.csv", index=False)
y_train.to_csv("y_train_blood.csv", index=False)
x_test.to_csv("test_blood.csv", index=False)
y_test.to_csv("y_test_blood.csv", index=False)
