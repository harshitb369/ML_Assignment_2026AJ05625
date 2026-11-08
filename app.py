import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide", page_icon="🛎️")

st.title("🛎️ Hospitality Analytics: Cancellation Prediction")
st.markdown("Upload a test dataset to evaluate custom-trained classification algorithms.")

st.sidebar.header("System Settings")
data_upload = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")

model_registry = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "K-Nearest Neighbors": "K_Nearest_Neighbors.pkl",
    "Gaussian Naive Bayes": "Naive_Bayes.pkl",
    "Support Vector Machine": "SVM_Classifier.pkl"
}
selected_algorithm = st.sidebar.selectbox("Select ML Classifier", list(model_registry.keys()))

if data_upload is not None:
    input_dataframe = pd.read_csv(data_upload)
    
    if 'is_canceled' not in input_dataframe.columns:
        st.error("Validation Error: The dataset must contain the 'is_canceled' target column.")
    else:
        input_dataframe.fillna({'agent': 0, 'company': 0, 'children': 0, 'country': 'Unknown'}, inplace=True)
        
        category_cols = ['hotel', 'meal', 'market_segment', 'deposit_type', 'customer_type']
        numeric_cols = [
            'lead_time', 'previous_cancellations', 'booking_changes', 'adr', 
            'total_of_special_requests', 'required_car_parking_spaces', 
            'adults', 'stays_in_week_nights', 'stays_in_weekend_nights'
        ]
        
        true_labels = input_dataframe['is_canceled']
        raw_features = input_dataframe[numeric_cols + category_cols]
        
        try:
            transformer_pipeline = joblib.load('model/data_transformer.pkl')
            processed_features = transformer_pipeline.transform(raw_features)
            
            active_model = joblib.load(f"model/{model_registry[selected_algorithm]}")
            
            predictions = active_model.predict(processed_features)
            probabilities = active_model.predict_proba(processed_features)[:, 1]
            
            st.markdown("### Classifier Performance Metrics")
            panel1, panel2, panel3, panel4, panel5, panel6 = st.columns(6)
            
            panel1.metric("Accuracy", f"{accuracy_score(true_labels, predictions):.3f}")
            panel2.metric("AUC ROC", f"{roc_auc_score(true_labels, probabilities):.3f}")
            panel3.metric("Precision", f"{precision_score(true_labels, predictions):.3f}")
            panel4.metric("Recall", f"{recall_score(true_labels, predictions):.3f}")
            panel5.metric("F1 Score", f"{f1_score(true_labels, predictions):.3f}")
            panel6.metric("MCC", f"{matthews_corrcoef(true_labels, predictions):.3f}")
            
            st.divider()
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("#### Confusion Matrix")
                c_matrix = confusion_matrix(true_labels, predictions)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(c_matrix, annot=True, fmt='d', cmap='cividis', ax=ax_cm)
                ax_cm.set_xlabel('Predicted Cancellation')
                ax_cm.set_ylabel('Actual Cancellation')
                st.pyplot(fig_cm)
                
            with viz_col2:
                st.markdown("#### Classification Report")
                st.code(classification_report(true_labels, predictions))
                
        except FileNotFoundError as e:
            st.error(f"Missing file error. Ensure models and transformer are in 'model/'. Details: {e}")
        except Exception as e:
            st.error(f"An unexpected inference error occurred: {e}")
else:
    st.info("Awaiting Data: Please upload 'streamlit_test_data.csv' via the sidebar to begin evaluation.")
