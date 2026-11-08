"""
Hotel Booking Cancellation - Machine Learning Training Pipeline
Executes EDA, preprocessing (Imputation, One-Hot Encoding, Scaling, Target Encoding), 
and hyperparameter-tuned model training for optimal Accuracy, Precision, and Recall.
"""
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)

# ==========================================
# Setup
# ==========================================
model_export_path = 'model'
visualization_path = 'exploratory_charts'
os.makedirs(model_export_path, exist_ok=True)
os.makedirs(visualization_path, exist_ok=True)

# ==========================================
# 2. Data loading and EDA
# ==========================================
hospitality_data = pd.read_csv('hotel_bookings.csv')

print("Generating Exploratory Data Analysis (EDA) visuals...")
plt.figure(figsize=(7,5))
sns.countplot(data=hospitality_data, x='is_canceled', palette='viridis')
plt.title('Booking Cancellation Distribution')
plt.savefig(f'{visualization_path}/cancellation_distribution.png')
plt.close()

numeric_subset = hospitality_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_subset.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='YlGnBu')
plt.title('Feature Correlation Heatmap')
plt.savefig(f'{visualization_path}/numeric_correlation_map.png')
plt.close()

# ==========================================
# Data Cleansing and Imputation
# ==========================================
missing_val_strategy = {'agent': 0, 'company': 0, 'children': 0, 'country': 'Unknown'}
hospitality_data.fillna(missing_val_strategy, inplace=True)

# ==========================================
# Feature Selection and Encoding
# ==========================================
target_label = 'is_canceled'

category_cols = ['hotel', 'meal', 'market_segment', 'deposit_type', 'customer_type']
numeric_cols = [
    'lead_time', 'previous_cancellations', 'booking_changes', 'adr', 
    'total_of_special_requests', 'required_car_parking_spaces', 
    'adults', 'stays_in_week_nights', 'stays_in_weekend_nights'
]

selected_features = numeric_cols + category_cols
X_predictors = hospitality_data[selected_features]

label_encoder = LabelEncoder()
y_target = label_encoder.fit_transform(hospitality_data[target_label])

# ==========================================
# Train Test Split
# ==========================================
X_train, X_eval, y_train, y_eval = train_test_split(
    X_predictors, y_target, test_size=0.25, random_state=777
)

streamlit_eval_df = X_eval.copy()
streamlit_eval_df[target_label] = y_eval
streamlit_eval_df.sample(n=900, random_state=777).to_csv('streamlit_test_data.csv', index=False)
print("Exported 'streamlit_test_data.csv' for web deployment.")

# ==========================================
# Pipelines
# ==========================================
pipeline_transformer = ColumnTransformer(
    transformers=[
        ('scale_numbers', StandardScaler(), numeric_cols),
        ('encode_categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), category_cols)
    ]
)

X_train_transformed = pipeline_transformer.fit_transform(X_train)
X_eval_transformed = pipeline_transformer.transform(X_eval)

joblib.dump(pipeline_transformer, f'{model_export_path}/data_transformer.pkl')

# ==========================================
# Hyper Parameter Tuning
# ==========================================
# Adjusted hyperparameters to maximize all evaluation metrics simultaneously
algorithm_suite = {
    "Logistic_Regression": LogisticRegression(max_iter=2000, random_state=777),
    "Decision_Tree": DecisionTreeClassifier(max_depth=18, min_samples_split=10, random_state=777),
    "K_Nearest_Neighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Naive_Bayes": GaussianNB(),
    "SVM_Classifier": SVC(probability=True, random_state=777, max_iter=8000, C=2.0)
}

evaluation_results = []

for algo_name, algo_instance in algorithm_suite.items():
    print(f"Executing training phase for: {algo_name}")
    algo_instance.fit(X_train_transformed, y_train)
    
    joblib.dump(algo_instance, f'{model_export_path}/{algo_name}.pkl')
    
    predicted_classes = algo_instance.predict(X_eval_transformed)
    predicted_probabilities = algo_instance.predict_proba(X_eval_transformed)[:, 1]
    
    evaluation_results.append({
        "Model": algo_name,
        "Accuracy": accuracy_score(y_eval, predicted_classes),
        "AUC Score": roc_auc_score(y_eval, predicted_probabilities),
        "Precision": precision_score(y_eval, predicted_classes),
        "Recall": recall_score(y_eval, predicted_classes),
        "F1 Score": f1_score(y_eval, predicted_classes),
        "MCC": matthews_corrcoef(y_eval, predicted_classes)
    })

# ==========================================
# Results
# ==========================================
final_metrics_df = pd.DataFrame(evaluation_results).round(4)
print("\n--- PERFORMANCE METRICS FOR README.MD ---")
print(final_metrics_df.to_markdown(index=False))