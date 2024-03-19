import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

os.chdir('C://Users//danie//Documents//GitHub//Machine_Learning_Daniel_Claesson//Lab')
X_test = pd.read_csv('Lab_1_test_samples.csv')
my_model = joblib.load('Lab_1_production_model_forest.joblib')
prediction = my_model.predict(X_test)
df = pd.DataFrame(prediction, columns=['prediction'])
df.to_csv('Lab_1_predictions.csv', index=False)