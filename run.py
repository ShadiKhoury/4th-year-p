#from interpretation import interpretation
#import pandas as pd
#interpretation("train_data.csv",test_data="test_data.csv",
#trian_labels="train_labels.csv",
#test_labels="test_labels.csv",model="lgbm",feature_imprtance_type="All_normilaze")
import interpretation
import sys
import os
#command = 'python interpretation.py --train_data "train_data.csv" --test_data "test_data.csv" --train_labels "train_labels.csv" --test_labels "test_labels.csv" --model "lgbm" --feature_imprtance_type "All_normilaze"'
#os.system(command)

command='python interpretation.py --train_data "train_blood.csv" --test_data "test_blood.csv" --train_labels "y_train_blood.csv" --test_labels "y_test_blood.csv" --model "lgbm_blood" --feature_imprtance_type "All_normilaze"'
os.system(command)
#python interpretation.py --train_data "train_data.csv" --test_data "test_data.csv" --train_labels "train_labels.csv" --test_labels "test_labels.csv" --model "lgbm" --feature_imprtance_type "All_normilaze"