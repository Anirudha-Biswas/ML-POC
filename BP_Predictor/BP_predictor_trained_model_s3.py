import pandas as pd
import joblib
import boto3
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data from S3
bucket = "inputfilesbucketcsv"
file_key = "bp_dataset.csv"

s3 = boto3.client("s3")                                                 
csv_obj = s3.get_object(Bucket=bucket, Key=file_key)
body = csv_obj['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(body))

# Train the model
X = df[['Age', 'Weight', 'Lifestyle']]
y = df['BloodPressure']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), ['Lifestyle'])
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)
joblib.dump(model, 'bp_model_v1.pkl')
print("Model trained and saved as bp_model_v1.pkl")
