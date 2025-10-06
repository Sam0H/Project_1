#clear console
#from IPython import get_ipython; get_ipython().magic('clear')

#importing and reading the data from excel file
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



data = pd.read_csv("data/Project_1_Data.csv")

# #drop rows with no data
# #data = data.dropna().reset_index(drop=True)

# # Quick look at the data 
# print("First 5 rows of the dataset:") 
# print(data.head())
# print("\nDataset info:")
# print(data.info())
# print("\nSummary statistics:")
# print(data.describe())
# # Check for missing values
# print("\nMissing values per column:")
# print(data.isnull().sum())

# Histograms for each variable
data.hist(bins=30, figsize=(10, 6))
plt.suptitle("Histograms of Features", fontsize=14)
plt.show()

# Boxplots of X, Y, Z grouped by Step
plt.figure(figsize=(12, 6))
data.boxplot(column=["X", "Y", "Z"], by="Step", figsize=(12, 6))
plt.suptitle("Boxplots of Features Grouped by Step")
plt.xlabel("Step")
plt.ylabel("Values")
plt.show()

# Scatter plot of X vs Y, colored by Step
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data["X"], data["Y"], c=data["Step"], cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Step")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter plot of X vs Y by Step")
plt.show()

# Scatter plot of X vs Z, colored by Step
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data["X"], data["Z"], c=data["Step"], cmap="plasma", alpha=0.7)
plt.colorbar(scatter, label="Step")
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Scatter plot of X vs Z by Step")
plt.show()

# Scatter plot of Y vs Z, colored by Step
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data["Y"], data["Z"], c=data["Step"], cmap="coolwarm", alpha=0.7)
plt.colorbar(scatter, label="Step")
plt.xlabel("Y")
plt.ylabel("Z")
plt.title("Scatter plot of Y vs Z by Step")
plt.show()






# Compute Pearson correlation
corr_matrix = data.corr(method="pearson")
print("\nCorrelation matrix:")
print(corr_matrix)

# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap of Features")
plt.show()





from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Split features and target
X = data.drop("Step", axis=1)
y = data["Step"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# --------------------------------
# Model 1: Logistic Regression
# --------------------------------
log_reg = LogisticRegression(max_iter=2000)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)

print("\nBest Logistic Regression Parameters:", grid_lr.best_params_)
print("Train Accuracy:", grid_lr.score(X_train, y_train))
print("Test Accuracy:", grid_lr.score(X_test, y_test))


# --------------------------------
# Model 2: Random Forest
# --------------------------------
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

print("\nBest Random Forest Parameters:", grid_rf.best_params_)
print("Train Accuracy:", grid_rf.score(X_train, y_train))
print("Test Accuracy:", grid_rf.score(X_test, y_test))


# --------------------------------
# Model 3: Support Vector Machine (SVM)
# --------------------------------
svm = SVC()
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_svm.fit(X_train, y_train)

print("\nBest SVM Parameters:", grid_svm.best_params_)
print("Train Accuracy:", grid_svm.score(X_train, y_train))
print("Test Accuracy:", grid_svm.score(X_test, y_test))


# --------------------------------
# Model 4: Gradient Boosting with RandomizedSearchCV
# --------------------------------
gb = GradientBoostingClassifier(random_state=42)
param_dist_gb = {
    'n_estimators': np.arange(50, 201, 50),
    'learning_rate': np.linspace(0.01, 0.3, 5),
    'max_depth': np.arange(2, 6),
    'min_samples_split': np.arange(2, 10)
}

random_search_gb = RandomizedSearchCV(
    gb, param_distributions=param_dist_gb, n_iter=20,
    cv=5, scoring='accuracy', random_state=42
)
random_search_gb.fit(X_train, y_train)

print("\nBest Gradient Boosting Parameters (RandomizedSearchCV):", random_search_gb.best_params_)
print("Train Accuracy:", random_search_gb.score(X_train, y_train))
print("Test Accuracy:", random_search_gb.score(X_test, y_test))




# =========================
# ~~~~~~~~~Step 5~~~~~~~~~~
# =========================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Train and Test Metrics
    print(f"\n{name} Performance Metrics:")
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Test Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Test Recall:", recall_score(y_test, y_pred_test, average='weighted'))
    print("Test F1 Score:", f1_score(y_test, y_pred_test, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

    return y_pred_test

# Evaluate all models
y_pred_lr = evaluate_model(grid_lr.best_estimator_, X_train, y_train, X_test, y_test, "Logistic Regression")
y_pred_rf = evaluate_model(grid_rf.best_estimator_, X_train, y_train, X_test, y_test, "Random Forest")
y_pred_svm = evaluate_model(grid_svm.best_estimator_, X_train, y_train, X_test, y_test, "SVM")

# Choose best model based on F1 (usually the most balanced metric)
best_model = grid_svm.best_estimator_   # Replace with whichever has best F1 in your results
y_pred_best = y_pred_svm

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Best Model")
plt.show()


# ==========================
# Step 6: Stacked Model Performance Analysis
# ==========================
from sklearn.ensemble import StackingClassifier

# Define the base learners (use best estimators from your grid search)
estimators = [
    ('rf', grid_rf.best_estimator_),
    ('svm', grid_svm.best_estimator_)
]

# Define the stacking classifier with Logistic Regression as the final estimator
stacked_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500),
    passthrough=False,  # set True if you want to include original features as well
    n_jobs=-1
)

# Fit the stacked model
stacked_clf.fit(X_train, y_train)

# Predictions
y_pred_stack = stacked_clf.predict(X_test)

# Evaluate stacked model
print("\nStacked Model Performance Metrics:")
print("Test Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Test Precision:", precision_score(y_test, y_pred_stack, average='weighted'))
print("Test Recall:", recall_score(y_test, y_pred_stack, average='weighted'))
print("Test F1 Score:", f1_score(y_test, y_pred_stack, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_stack))

# Confusion Matrix
cm_stack = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(6,4))
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Greens", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stacked Model")
plt.show()



# =============================
# Step 7: Model Evaluation
# =============================

import joblib

# Save model to disk
joblib.dump(best_model, "best_model.joblib")
print("Best model saved as best_model.joblib")

# Load model (to demonstrate reusability)
loaded_model = joblib.load("best_model.joblib")

# Predict maintenance steps for given coordinates
test_coords = np.array([
    [9.375, 3.0625, 1.51], 
    [6.995, 5.125, 0.3875], 
    [0, 3.0625, 1.93], 
    [9.4, 3, 1.8], 
    [9.4, 3, 1.3]
])

predictions = loaded_model.predict(test_coords)
print("\nPredictions for given coordinates:")
for coords, pred in zip(test_coords, predictions):
    print(f"Coordinates {coords} -> Predicted Step: {pred}")
