#from interpretation import interpretation
#import pandas as pd
#interpretation("train_data.csv",test_data="test_data.csv",c
#trian_labels="train_labels.csv",
#test_labels="test_labels.csv",model="lgbm",feature_imprtance_type="All_normilaze")
import interpretation
import sys
import os
#command = 'python interpretation.py --train_data "train_data.csv" --test_data "test_data.csv" --train_labels "train_labels.csv" --test_labels "test_labels.csv" --model "lgbm" --feature_imprtance_type "All_normilaze"'
#os.system(command)
#os.makedirs(r'C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\Data\.')
sys.path.insert(0,r'C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\Data\.')
#command='python interpretation.py --train_data "train_blood.csv" --test_data "test_blood.csv" --train_labels "y_train_blood.csv" --test_labels "y_test_blood.csv" --model "lgbm_blood" --feature_imprtance_type "All_normilaze"'

command=r'python interpretation.py --train_data "C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\Data\train_data.csv" --test_data "C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\Data\test_data.csv" --train_labels "C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\Data\train_labels.csv" --test_labels "C:\Users\shade\Desktop\עבודות\שנה ד״\פרויקט\VS_project\4th-year-project\Data\test_labels.csv" --model "lgbm" --feature_imprtance_type "All_normilaze"'
os.system(command)
