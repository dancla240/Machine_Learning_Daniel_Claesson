import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X_test = pd.read_csv("Lab_1_test_saples.csv")
my_model = joblib.load('Lab_1_production_model_svc.pkl')
my_model.predict(X_test)
print(my_model)

