import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('train_data.csv')

# Define features and target
X = df.drop('stress_level', axis=1)
y = df['stress_level']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the SVM model on the full dataset
svm_model = SVC(random_state=42)
svm_model.fit(X_scaled, y)

# Save the trained model and the scaler to files
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")