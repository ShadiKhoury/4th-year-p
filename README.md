![Tel_Aviv_university_logo_-_English](https://user-images.githubusercontent.com/88155916/171557623-1c7b3463-e9f6-48b8-9011-21443b542d9e.png)

# 4'th year Grad project in Bc.s in biomedical engineering
my 4th year project for Bc.s in biomedical engineering

### Interpretation
the main objective of this project os to validate diffrent feature impoertnace methods and understand thier diffrences on Real world Clincial data 
# How dose the code work ?

To run the Script use Interpoation.py 
where the args are : 

1 ) '--train_data'
2)'--test_data'
3)'--train_labels'
4)'--test_labels'
5)'--model'
6)'--feature_imprtance_type'
# Imports
<
parser = argparse.ArgumentParser(description='Import Data/model For Testing')
parser.add_argument('--train_data', metavar='TrainD',
                    help='Input Train Data')
parser.add_argument('--test_data', metavar='TestD',
                    help='Input Test Data')
parser.add_argument('--train_labels', metavar='TrainL',
                    help='Input Train Label')
parser.add_argument('--test_labels', metavar='TestL',
                    help='Input Test label')
parser.add_argument('--model', metavar='Model',type = str,
                    help='Input Moudle Name')
parser.add_argument('--feature_imprtance_type', metavar='importance_type',type = str,
                    help='Input importance type')
args = parser.parse_args()

>

and in turn this script predoces a jason file and a Barh plot with the normlized impoertance Scores for each feature for each method
