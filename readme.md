Mental Health Risk Prediction using AdaBoost
📌 Overview
This project implements an AdaBoost Classifier to predict mental health risk levels based on demographic, lifestyle, and psychosocial factors. The model uses a Decision Tree as the base estimator and achieves 82.18% accuracy with balanced class handling.

📊 Dataset
The dataset contains features related to mental health indicators and demographic information. The target variable is mental_health_risk with 3 classes:

Label	Risk Level
0	Low Risk
1	Moderate Risk
2	High Risk
Dataset Statistics
Split	Samples
Train	80%
Test	20%
Total	20,501 rows

🏗️ Model Architecture

AdaBoost with Decision Tree Stumps

AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
    n_estimators=200,
    learning_rate=0.01,
    random_state=42
)

Key Components
Component	Description
Base Estimator	Decision Tree with max_depth=3 (stump)
Class Weight	balanced - automatically adjusts weights for imbalanced classes
Ensemble Size	200 estimators
Learning Rate	0.01 (slower, more robust learning)
Loss Function	SAMME (Stagewise Additive Modeling using Multi-class Exponential loss)
How AdaBoost Works
Train weak learners (decision stumps) sequentially

Each subsequent model focuses on misclassified samples

Weighted voting combines predictions from all models

Lower learning rate = smaller contribution per model, requiring more estimators

🚀 Training Configuration
Data Preprocessing
Step	Method
Missing Values	Median imputation for numeric columns
Categorical Encoding	Label Encoding (preserved in pipeline)
Feature Scaling	StandardScaler (standardization)
Train-Test Split	80/20 with stratification
Hyperparameters
Parameter	Value	Reason
max_depth	3	Prevents overfitting, creates weak learners
class_weight	'balanced'	Handles class imbalance automatically
n_estimators	200	Sufficient for convergence with low LR
learning_rate	0.01	Smaller steps = better generalization
📈 Performance Results
Overall Metrics
Metric	Score
Accuracy	82.18%
Classification Report
Class (Risk Level)	Precision	Recall	F1-Score	Support
0 (Low)	0.94	0.99	0.96	1,871
1 (Moderate)	0.75	0.95	0.83	2,365
2 (High)	1.00	0.03	0.06	764
Overall Averages
Avg Type	Precision	Recall	F1-Score
Macro Avg	0.89	0.65	0.62
Weighted Avg	0.86	0.82	0.76
Key Observations
✅ Excellent performance on Class 0 (Low Risk)

High precision (0.94) and recall (0.99)

✅ Good performance on Class 1 (Moderate Risk)

Strong recall (0.95) indicating few false negatives

⚠️ Poor performance on Class 2 (High Risk)

Very low recall (0.03) - only 3% of high-risk cases identified

This is a class imbalance issue (only 764 samples vs. 4,236 for other classes)

📁 Project Structure

mental_health_prediction/
├── Final.csv                              # Original dataset
├── mental_health_adaboost_model.pkl       # Saved model + preprocessors
├── row_data/                              # Individual row CSVs (20,501 files)
│   ├── row_1.csv
│   ├── row_2.csv
│   └── ...
├── confusion_matrix.png                   # Confusion matrix visualization
└── training.ipynb                         # Main training notebook

Saved Model Components
The pickle file mental_health_adaboost_model.pkl contains:

{
    'model': ada_model,              # Trained AdaBoost classifier
    'scaler': scaler,                # StandardScaler for features
    'le_target': le_target,          # LabelEncoder for target
    'encoders': encoders,            # Dict of encoders for categorical cols
    'feature_columns': feature_columns # List of feature names
}

🛠️ Requirements

# Core Dependencies
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
pickle (built-in)

Installation

pip install pandas numpy scikit-learn matplotlib seaborn

📖 Usage
1. Training the Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv('Final.csv')

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
encoders = {}

for col in categorical_cols:
    if col != 'mental_health_risk':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Encode target
le_target = LabelEncoder()
df['mental_health_risk'] = le_target.fit_transform(df['mental_health_risk'].astype(str))

# Prepare features
X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train AdaBoost
base_learner = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
ada_model = AdaBoostClassifier(
    estimator=base_learner,
    n_estimators=200,
    learning_rate=0.01,
    random_state=42
)
ada_model.fit(X_train, y_train)


2. Making Predictions on a Single Row

import pandas as pd
import pickle

# Load model
with open('mental_health_adaboost_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
le_target = model_data['le_target']
encoders = model_data['encoders']
feature_columns = model_data['feature_columns']

# Load input data (single row CSV)
df = pd.read_csv('row_data/row_6583.csv')

# Apply encoding
for col, le in encoders.items():
    if col in df.columns:
        df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Ensure all features exist
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0

df = df[feature_columns]

# Scale and predict
X_scaled = scaler.transform(df)
prediction = model.predict(X_scaled)
predicted_label = le_target.inverse_transform(prediction)

print(f"Predicted Mental Health Risk: {predicted_label[0]}")

3. Batch Processing (Export Rows)

# Save each row as separate CSV for individual testing
df = pd.read_csv('Final.csv')
df = df.drop('mental_health_risk', axis=1)

output_folder = "row_data"
os.makedirs(output_folder, exist_ok=True)

for i, row in df.iterrows():
    row_df = pd.DataFrame([row])
    row_df.to_csv(f"{output_folder}/row_{i+1}.csv", index=False)

📊 Visualizations
Confusion Matrix
The notebook generates a confusion matrix heatmap showing:

True Positives (diagonal)

False Positives (off-diagonal)

Class distribution across predictions

plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

🔧 Model Optimization Suggestions
To improve Class 2 (High Risk) detection:

Technique	Implementation
SMOTE	Synthetic oversampling of minority class
Class Weights	Manually increase weight for class 2
Threshold Tuning	Lower decision threshold for class 2
Collect More Data	Additional high-risk samples
Alternative Models	Random Forest, XGBoost
Example: SMOTE Implementation

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train on resampled data
ada_model.fit(X_train_resampled, y_train_resampled)

🎯 Advantages of AdaBoost for Mental Health Prediction
Advantage	Benefit
Ensemble Method	Reduces overfitting compared to single decision tree
Sequential Learning	Focuses on difficult-to-classify cases
Interpretable	Feature importance can be extracted
Handles Mixed Data	Works with both numeric and categorical features
Robust	Less sensitive to outliers than other algorithms
📁 Output Files
File	Description
mental_health_adaboost_model.pkl	Serialized model + preprocessors
confusion_matrix.png	Confusion matrix visualization
row_data/*.csv	Individual row files (20,501 files)
📊 Feature Importance Analysis
To extract feature importance from the trained model:

# Get feature importance from AdaBoost
feature_importance = model.feature_importances_

# Sort by importance
sorted_idx = np.argsort(feature_importance)[::-1]

print("Top 10 Most Important Features:")
for i in range(min(10, len(feature_columns))):
    print(f"{feature_columns[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")


🔄 Model Pipeline Visualization

Raw Input (CSV row)
        ↓
Label Encoding (categorical features)
        ↓
Feature Alignment (ensure all columns exist)
        ↓
Standard Scaling (mean=0, std=1)
        ↓
AdaBoost Ensemble (200 Decision Tree stumps)
        ↓
Weighted Voting
        ↓
Prediction (0, 1, or 2)
        ↓
Inverse Label Encoding
        ↓
Risk Level Output

👥 Contributors
Author: [Your Name]

Framework: scikit-learn

Algorithm: AdaBoost with Decision Tree base estimator

📚 References
Freund, Y., & Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. Journal of Computer and System Sciences.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.

Zhu, J., Zou, H., Rosset, S., & Hastie, T. (2009). Multi-class AdaBoost. Statistics and Its Interface.

📄 License
This project is for research and educational purposes.

⚠️ Disclaimer
This model is intended for research and educational purposes only. It should not be used as a substitute for professional mental health assessment, diagnosis, or treatment. Always consult qualified healthcare professionals for mental health concerns.


