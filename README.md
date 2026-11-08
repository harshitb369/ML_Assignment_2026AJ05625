# Hotel Booking Cancellation ML Project

## a. Problem Statement
The objective of this assignment is to develop and evaluate multiple machine learning classifiers to predict hotel booking cancellations. Accurately forecasting cancellations allows hospitality businesses to optimize their inventory and reduce revenue loss.

## b. Dataset Description
The dataset used is `hotel_bookings.csv` containing over 119,000 instances. For this binary classification problem, 14 numerical and categorical features (such as lead time, special requests, and previous cancellations) were selected to predict the target variable `is_canceled`.

## c. Github Repository Link
https://github.com/harshitb369/ML_Assignment_2026AJ05625

## d. Models Used & Comparison

| Model               |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |    MCC |
|:--------------------|-----------:|------------:|------------:|---------:|-----------:|-------:|
| Logistic_Regression |     0.8065 |      0.853  |      0.8395 |   0.5983 |     0.6987 | 0.5794 |
| Decision_Tree       |     0.8278 |      0.8834 |      0.8182 |   0.6952 |     0.7517 | 0.626  |
| K_Nearest_Neighbors |     0.8337 |      0.8878 |      0.8003 |   0.7416 |     0.7698 | 0.6411 |
| Naive_Bayes         |     0.6796 |      0.8041 |      0.5509 |   0.7879 |     0.6484 | 0.3907 |
| SVM_Classifier      |     0.5769 |      0.6845 |      0.4625 |   0.7913 |     0.5838 | 0.2418 |

### Observations
ML Model Name       |       Observation about model performance
Logistic Regression | "Served as a highly stable linear baseline with strong Precision (83.9%), but struggled with Recall (59.8%),      meaning it successfully identified obvious cancellations but missed more complex edge cases."

Decision Tree       | "Captured non-linear booking patterns beautifully. Thanks to hyperparameter tuning (max_depth and min_samples_split), it avoided severe overfitting and achieved a strong balance of Precision and Recall (F1: 0.751)."

KNN                 |   "The best performing model overall. By using distance-weighting on the scaled features, it successfully grouped similar booking behaviors together, yielding the highest F1 Score (0.769) and MCC (0.641)."

Naive Bayes         | "Achieved high Recall but very poor Precision (55.0%). Its fundamental assumption that all features are independent failed here, as real-world booking features (e.g., adults and market_segment) are highly correlated, leading to many false alarms."

Support Vector Machine | "Yielded the lowest overall metrics. Because the dataset was very large and highly dimensional (due to One-Hot Encoding), the SVM struggled to converge. The iterations had to be mathematically capped to prevent system timeouts, making it unscalable for this specific dataset."

**Overall Winner for your dataset?** 
Based on the metrics above, the K-Nearest Neighbors (KNN) is the overall winner because it achieves the highest combined F1 and MCC score, perfectly balancing precision and recall for this specific cancellation problem.
