from joblib import load
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

os.chdir('C://Users//danie//Documents//GitHub//Machine_Learning_Daniel_Claesson//Lab')
test_samples = pd.read_csv('Lab_1_test_samples.csv')
X_test = test_samples.drop(['cardio'], axis=1)

my_model = load('Lab_1_production_model_forest.joblib')
prediction = my_model.predict(X_test)
probability = my_model.predict_proba(X_test)

df = pd.DataFrame(prediction, columns=['prediction'])
df['probability class 0'] = pd.DataFrame(probability[:,0])
df['probability class 1'] = pd.DataFrame(probability[:,1])

df.to_csv('Lab_1_predictions.csv', index=False)