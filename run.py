from interpretation import interpretation
import pandas as pd
interpretation(trian_data="train_data.csv",test_data="test_data.csv",
trian_labels="train_labels.csv",
test_labels="test_labels.csv",model="lgbm",feature_imprtance_type="dice_local_cf")
