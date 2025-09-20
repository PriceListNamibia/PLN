import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Expanded Medicaid applications dataset
data = {
    'applicant_id': range(1, 21),
    'income': np.random.randint(5000, 50000, 20),
    'age': np.random.randint(18, 80, 20),
    'employment_status': np.random.choice([0, 1], 20),  # 1=employed, 0=unemployed
    'has_dependents': np.random.choice([0, 1], 20),
    'disability_status': np.random.choice([0, 1], 20),  # 1=disabled, 0=not disabled
    'prior_claims': np.random.randint(0, 5, 20),
    'flagged_fraud': np.random.choice([0, 1], 20, p=[0.8, 0.2])  # 1=fraud, 0=clean
}

df = pd.DataFrame(data)

# Feature engineering: income per dependent
df['income_per_dependent'] = df.apply(lambda row: row['income'] / (row['has_dependents'] + 1), axis=1)

# Features for model
features = ['income', 'age', 'employment_status', 'has_dependents', 'disability_status', 'prior_claims', 'income_per_dependent']
X = df[features]
y = df['flagged_fraud']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Predictions
predictions = clf.predict(X_test)

# Results
results = X_test.copy()
results['Predicted_Fraud'] = predictions
results['Actual_Fraud'] = y_test.values

print("Medicaid Application Fraud Simulation Results (Enhanced):")
print(results)

# Eligibility check: income < $30,000, age < 65, not flagged for fraud, and not more than 2 prior claims
df['eligible'] = df.apply(lambda row: 1 if (row['income'] < 30000 and row['age'] < 65 and row['flagged_fraud'] == 0 and row['prior_claims'] <= 2) else 0, axis=1)
print("\nEligibility Results:")
print(df[['applicant_id', 'income', 'age', 'prior_claims', 'flagged_fraud', 'eligible']])

# Model evaluation
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# Feature importance
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance_df)

# Visualization: eligibility distribution
plt.figure(figsize=(6, 4))
df['eligible'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Eligibility Distribution')
plt.xticks([0, 1], ['Not Eligible', 'Eligible'], rotation=0)
plt.ylabel('Number of Applicants')
plt.tight_layout()
plt.show()

# Visualization: fraud prediction rates
plt.figure(figsize=(6, 4))
results['Predicted_Fraud'].value_counts().plot(kind='bar', color=['lightgreen', 'tomato'])
plt.title('Fraud Prediction Distribution')
plt.xticks([0, 1], ['Clean', 'Fraud'], rotation=0)
plt.ylabel('Number of Applicants (Test Set)')
plt.tight_layout()
plt.show()