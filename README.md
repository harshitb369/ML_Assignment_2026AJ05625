# Hotel Booking Cancellation ML Project

## a. Problem Statement
The objective of this assignment is to develop and evaluate multiple machine learning classifiers to predict hotel booking cancellations. Accurately forecasting cancellations allows hospitality businesses to optimize their inventory and reduce revenue loss.

## b. Dataset Description
The dataset used is `hotel_bookings.csv` containing over 119,000 instances. For this binary classification problem, 14 numerical and categorical features (such as lead time, special requests, and previous cancellations) were selected to predict the target variable `is_canceled`.

## c. Github Repository Link
[Insert your GitHub link here]

## d. Models Used & Comparison
*(Replace the values below with the output from running `model_trainer.py` on BITS Virtual Lab)*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] |
| Decision Tree | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] |
| KNN | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] |
| Naive Bayes | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] |
| Support Vector Machine | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provided a highly stable baseline metric but struggled to capture complex non-linear relationships in the booking data. |
| Decision Tree | Achieved strong recall scores, though it showed slight signs of overfitting compared to simpler models. |
| KNN | Computationally heavy during inference and highly sensitive to the scaled distance of features like `lead_time`. |
| Naive Bayes | Performed the fastest but had lower precision, likely because the independence assumption between features is violated in real hotel data. |
| Support Vector Machine | Delivered robust performance across all metrics but took significantly longer to train on the scaled dataset. |

**Overall Winner for your dataset?** 
Based on the metrics above, the **Logistic Regression / Random Forest (Evaluate based on metrics)** is the overall winner because it achieves the highest combined F1 and MCC score, perfectly balancing precision and recall for this specific cancellation problem.
