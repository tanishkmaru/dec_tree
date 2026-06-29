Medicine Category Recommendation System using Decision Trees

Overview

This project is a Machine Learning-based Medicine Recommendation System that predicts an appropriate medicine based on a patient's symptoms.

The system uses Decision Tree Classification to learn relationships between symptoms and medicine classes. It compares both a depth-limited decision tree and an unlimited decision tree to evaluate their performance.

The project also provides an interactive symptom predictor where users can enter symptoms and view the decision-making process of the model.

This project is intended for educational purposes only and should not be used as a substitute for professional medical advice.


Features

- Uses two medical datasets for prediction
- Data cleaning and preprocessing
- One-hot encoding of symptoms
- Disease-to-medicine mapping
- Trains both depth-limited and unlimited Decision Tree models
- Compares model performance using:
  - Training Accuracy
  - Testing Accuracy
  - Cross Validation
- Generates classification reports
- Displays confusion matrices
- Shows feature importance
- Visualizes decision trees
- Interactive symptom-based prediction
- Displays the decision path followed by the model

---

## Datasets Used

### 1. dataset.csv

Contains:
- Disease names
- Up to 17 symptoms for each disease

### 2. Symptom-severity.csv

Contains:
- Symptom names
- Severity values for each symptom

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Machine Learning Workflow

1. Load the datasets
2. Clean symptom and disease names
3. Convert symptoms into one-hot encoded features
4. Map diseases to medicine categories
5. Split the dataset into training and testing sets
6. Train Decision Tree models
7. Evaluate model performance
8. Generate visualizations
9. Predict medicines based on user-entered symptoms

---

## Evaluation Metrics

The models are evaluated using:

- Training Accuracy
- Testing Accuracy
- 5-Fold Cross Validation
- Classification Report
- Confusion Matrix

---

## Visualizations

The project generates the following graphs:

- Accuracy vs Tree Depth
- Medicine Distribution
- Feature Importance
- Confusion Matrices
- Decision Tree Visualization
- Decision Path Visualization

---

## Installation

Clone the repository
Install dependencies

pip install pandas numpy matplotlib scikit-learn seaborn

Place the following files in the project directory:

dataset.csv
Symptom-severity.csv

Run the project

python medicine_recommendation.py
Example

Input Symptoms

fever
chills
vomiting
sweating
headache

Output

Recommended Medicine:
Antimalarial

The system also displays:

Confidence score
Top 3 medicine predictions
Decision path
Tree visualization


Project Structure
Medicine-Recommendation-System/
│
├── dataset.csv
├── Symptom-severity.csv
├── medicine_recommendation.py
├── depth_analysis.png
├── medicine_distribution.png
├── feature_importances.png
├── confusion_matrices.png
├── tree_limited.png
├── README.md


Future Improvements
GUI/Desktop interface
Web application deployment
Real medicine database integration
Probability-based disease prediction
Support for multiple disease predictions
Improved symptom matching using NLP


Disclaimer

This project is developed for educational and research purposes only. The predictions are generated using a machine learning model and should not be considered medical advice. Always consult a qualified healthcare professional before taking any medication.


