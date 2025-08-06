# Model-Results-and-Data-Splits
Reflection and code on Model Results and Data Splits
1. What impact does the 60:20:20 (from 75:15:15) split have on model accuracy?
When increasing the validation and test sizes from 15% to 20% each, the validation accuracy decreased from 96.3% to 94.4%. I believe the decrease in validation accuracy occurred because reducing the training set size gives the model less data to learn from, which can negatively affect its ability to generalize to unseen data. On the other hand the test accuracy increased from 96.3% to 97.2% as having a larger test set may provide a more stable and representative estimate of final model performance.
It’s important to highlight that the wine dataset contains only 178 rows, making it relatively small. As a result, changes in the train-validation-test split could lead to noticeable fluctuations in accuracy rates (i.e. with limited data, the choice of data split becomes even more important).
Additionally, it's worth noting that the logistic regression model did not fully converge after the default 1000 iterations, as indicated by a convergence warning. This may suggest the need to either increase the number of iterations.
2. How does the model's performance change if you use a 70:15:15 split and apply an SVM?
Applying an SVM classifier with the same 70:15:15 split resulted in a lower validation accuracy of ~74%, compared to the logistic regression model. This suggests that SVM may be overfitting or underfitting on this dataset. A simpler logistic regression appears better suited here. 
3. What might happen if you omit the validation set and only use training and testing data?
Omitting the validation set removes an independent dataset used for model selection. This can compromise the integrity and accuracy of the model because:
•	Test accuracy becomes unreliable: in the absence of a validation set, the test data may be inappropriately used for model tuning, introducing bias and undermining the credibility of performance evaluation and test results.
•	The risk of overfitting increases: the model may learn patterns specific to the training data without an intermediate check from the validation sample, leading to poor generalization to unseen data, which may result in lower test accuracy.
4. How can you apply what you’ve learned from experimenting with different data splits and model types to improve your capstone project’s model performance and reliability?
These experiments highlight the importance of using a validation set to fine-tune models, selecting appropriate data splits between training, validation and test (that may takes into account also the sample size) and of testing multiple model types, keeping in mind that increased complexity does not always lead to better accuracy, especially with limited data.
For my capstone project, I will ensure that I include a clear split between training, validation, and test datasets during the model development phase. I will also compare multiple model types. In addition to the approaches used in this exercise, I plan to explore different levels of parameterization for logistic regression. Finally, it is essential that the final model selection considers not only performance and accuracy but also the interpretability of the results.
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Load the default wine dataset
data = load_wine()
X = data.data
y = data.target


# Convert to a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column (wine class)
df['target'] = data.target

# Save to CSV
df.to_csv('wine_dataset.csv', index=False)

print("File saved as 'wine_dataset.csv'")

import os
print(os.getcwd())

print(y)
#print(X)

# Split data into train (70%), validation (15%) and test (15%)sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)
# Fit the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the validation set
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate the test set
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report on test set
print("\nClassification Report on Test Set:")
print(classification_report(y_test, test_preds, target_names=data.target_names))



# Split data into train (60%), validation (20%) and test (20%)sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)
# Fit the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the validation set
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate the test set
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report on test set
print("\nClassification Report on Test Set:")
print(classification_report(y_test, test_preds, target_names=data.target_names))

#How does the model's performance change if you use a 70:15:15 split and apply an SVM?
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val
)
# Initialize and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on validation set
val_preds = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)

# Print results
print("Validation Accuracy:", val_accuracy)
print("\nClassification Report (Validation Set):")
print(classification_report(y_val, val_preds, target_names=data.target_names))


# you omit the validation set and only use training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.70, random_state=42, stratify=y
)

# Fit the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the test set
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

