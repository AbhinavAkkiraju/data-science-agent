import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace_config import configure_trace_environment

configure_trace_environment()

from opto import trace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from opto.optimizers import OptoPrime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from opto.trace import Module

@trace.model
class SpaceshipTitanicPipeline:
    def __call__(self, x, y=None, test_data=None):
        prediction = self.train_model(x, y, test_data)
        return prediction
    
    @trace.bundle(trainable=True)
    def train_model(self, x, y=None, test_data=None):
        """
        Make predictions on whether passengers were transported for the Spaceship Titanic Kaggle Competition.

        Preprocessing Steps (some examples on how you could do this, however you can use your own method if it works better):
        1. Missing Value Handling:
            - Numerical features: Intelligent imputation (median, mean, or 0)
            - Categorical features: Mode filling or meaningful defaults
            - Outlier detection and treatment

        2. Feature Engineering:
            - Passenger ID parsing:
                * Extract group and individual identifiers
                * Create group-related features
            - Cabin information extraction:
                * Deck identification
                * Cabin number parsing
                * Side (port/starboard) classification
            - Name feature parsing:
                * Title extraction
                * Potential family relationship inference

        3. Advanced Feature Creation:
            - Family size computation
            - Total and relative spending calculations
            - Amenity usage patterns
            - Spatial features (cabin location metrics)

        4. Categorical Variable Handling:
            - One-hot encoding
            - Label encoding
            - Embedding techniques for high-cardinality features

        5. Numerical Feature Transformation:
            - Scaling (StandardScaler, MinMaxScaler)
            - Skewness correction (log, square root, Box-Cox)
            - Normalization techniques

        Selection Methodology (some examples on how you could do this, however you can use your own method if it works better):
        1. Statistical Feature Importance:
            - Correlation analysis
            - Mutual information
            - Chi-squared tests
            - Model-based feature importance

        2. Feature Weighting Criteria:
            - Predictive power for transportation status
            - Domain-specific relevance
            - Minimal multicollinearity
            - Computational efficiency

        3. Key Feature Categories:
            - Demographic Signals:
                * Age distribution
                * Planetary origin
            - Travel Characteristics:
                * Cabin attributes
                * CryoSleep status
                * VIP designation
            - Economic Indicators:
                * Spending patterns
                * Service utilization

        4. Selection Mechanism:
            - Probabilistic feature selection
            - Dynamic weight adjustment
            - Prevent overfitting through selective inclusion

        Ensemble Strategy (some examples on how you could do this, however you can use your own method if it works better):
        1. Model Diversity:
            - Tree-based models (Random Forest, Gradient Boosting)
            - Linear models (Logistic Regression variants)
            - Support Vector Machines
            - Probabilistic classifiers

        2. Ensemble Techniques:
            - Voting, boosting, bagging, stacking
            - Stacking with meta-learners
            - Weighted model combination
            - Regularization-aware model selection

        3. Hyperparameter Optimization:
            - Cross-validated parameter tuning
            - Regularization strength balancing
            - Learning rate and depth control
            - Subsample and feature sampling strategies

        4. Computational Considerations:
            - Computational complexity management
            - Memory-efficient model design
            - Scalable ensemble construction
        
        Training Methodology (some examples on how you could do this, however you can use your own method if it works better):
        1. Data Preparation:
            - Feature subset preparation
            - Cross-validation splitting
            - Stratified sampling

        2. Class Imbalance Handling:
            - Weighted loss functions
            - SMOTE oversampling
            - Synthetic data generation
            - Class-aware regularization

        3. Regularization Techniques:
            - L1/L2 penalty integration
            - Dropout-like regularization
            - Early stopping mechanisms
            - Gradient clipping

        4. Training Optimization:
            - Adaptive learning rates
            - Ensemble member performance tracking
            - Dynamic weight adjustment
            - Prediction confidence calibration

        Prediction Workflow (some examples on how you could do this, however you can use your own method if it works better):
        1. Probabilistic Prediction:
            - Soft classification probabilities
            - Confidence-based thresholding
            - Ensemble prediction aggregation

        2. Post-processing Techniques:
            - Calibration curves
            - Probability scaling
            - Ensemble diversity preservation

        3. Output Formatting:
            - Binary classification output
            - Kaggle submission compatibility
            - Interpretable prediction format

        4. Prediction Quality Assessment:
            - Uncertainty quantification
            - Prediction reliability scoring
            - Anomaly detection

        Args:
            x (pd.DataFrame): Input training or test data
            y (pd.Series, optional): Training labels
            test_data (pd.DataFrame, optional): Test dataset for final predictions
        
        Returns:
            Processed training predictions or test predictions
        """
        if test_data: return [0] * len(test_data)
        return [0] * len(y)

train_data = pd.read_csv(r'spaceship-titanic/data/train.csv')
test_data = pd.read_csv(r'spaceship-titanic/data/test.csv')

X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

agent = SpaceshipTitanicPipeline()
optimizer = OptoPrime(agent.parameters(), learning_rate=0.001, memory_size=5)

history = {"iteration": [], "train_accuracy": [], "val_accuracy": [], "val_precision": [], 
           "val_recall": [], "val_f1": []}
best_val_f1 = 0
best_model = None

epoch = 0
failed = False
while epoch < 20:
    try:
        train_predictions = agent(X_train, y_train)
        val_predictions = agent(X_train, y_train, X_val)
        train_accuracy = accuracy_score(y_train.values, train_predictions.data)
        if len(y_val) == len(val_predictions.data):
            failed = False
            val_accuracy = accuracy_score(y_val.values, val_predictions.data)
            val_precision = precision_score(y_val.values, val_predictions.data)
            val_recall = recall_score(y_val.values, val_predictions.data)
            val_f1 = f1_score(y_val.values, val_predictions.data)
        else:
            failed = True
            val_accuracy = 0.0000
            val_precision = 0.0000
            val_recall = 0.0000
            val_f1 = 0.0000
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for train_idx, val_idx in kf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_preds = agent(X_fold_train, y_fold_train, X_fold_val)
            
            cv_metrics['accuracy'].append(accuracy_score(y_fold_val, fold_preds.data))
            cv_metrics['precision'].append(precision_score(y_fold_val, fold_preds.data))
            cv_metrics['recall'].append(recall_score(y_fold_val, fold_preds.data))
            cv_metrics['f1'].append(f1_score(y_fold_val, fold_preds.data))

        cv_mean_accuracy = np.mean(cv_metrics['accuracy'])
        cv_mean_f1 = np.mean(cv_metrics['f1'])
        cv_mean_precision = np.mean(cv_metrics['precision'])
        cv_mean_recall = np.mean(cv_metrics['recall'])

        improving = ""
        if epoch > 0:
            if val_f1 > history["val_f1"][-1]:
                improving = "Good improvement! "
            elif val_f1 < history["val_f1"][-1]:
                improving = "Performance decreased. Try going back to what you had previously and then improving. "
            else: 
                improving = "Performance stayed the same. Try changing your model or features. "
        
        feedback = f"Epoch {epoch + 1}/{20}: {improving}Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}. "
        
        if failed:
            feedback += f"Warning: Mismatch in validation data lengths. y_val: {len(y_val)}, val_predictions: {len(val_predictions.data)}. "

        if val_f1 < 0.5:
            feedback += "Model performance is poor. Try better feature engineering and preprocessing. "
        elif val_f1 < 0.7:
            feedback += "Model is showing promise but needs improvement. Consider class balancing techniques. "
        elif val_f1 < 0.8:
            feedback += "Model is performing well. Fine-tune hyperparameters for further improvements. "
        else:
            feedback += "Excellent performance! Focus on preventing overfitting. "

        feedback += "You are trying to maximize F1 score to at least above 0.9. "
        
        history["iteration"].append(epoch + 1)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            import copy
            best_model = copy.deepcopy(agent)
            
        epoch += 1
        
    except trace.ExecutionError as e:
        val_predictions = e.exception_node
        feedback = f"Make sure to fix the following error in epoch {epoch + 1}: {str(e)}"
    
    print(feedback)
    
    optimizer.zero_feedback()
    optimizer.backward(val_predictions, feedback)
    optimizer.step(verbose=False)

agent = best_model
final_predictions = agent(X, y, test_data)
submission_df = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Transported": final_predictions.data
})

submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)

Module.save(self=best_model, file_name='agent.pkl')
print(f"Submission file saved at: {submission_path}")
print(f"Best validation F1 score: {best_val_f1:.4f}")

plt.figure(figsize=(12, 6))
epochs = history["iteration"]
plt.plot(epochs, history["train_accuracy"], label='Training Accuracy', marker='o', linestyle='-', color='blue')
plt.plot(epochs, history["val_accuracy"], label='Validation Accuracy', marker='s', linestyle='-', color='red')
plt.plot(epochs, history["val_f1"], label='Validation F1', marker='d', linestyle='-', color='green')
plt.axhline(y=0.8, color='purple', linestyle='--', label='Target F1 (0.8)')
plt.title('Model Performance Over Training Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
best_epoch = history["val_f1"].index(max(history["val_f1"])) + 1
best_f1 = max(history["val_f1"])
plt.annotate(f'Best F1: {best_f1:.4f}', 
             xy=(best_epoch, best_f1),
             xytext=(best_epoch+1, best_f1+0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)
if len(history["val_f1"]) > 3:
    last_three_avg = sum(history["val_f1"][-3:]) / 3
    first_three_avg = sum(history["val_f1"][:3]) / 3
    improvement = ((last_three_avg - first_three_avg) / first_three_avg) * 100
    plt.figtext(0.15, 0.02, f'Overall improvement: {improvement:.1f}%', fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('model_performance.pdf')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, history["val_precision"], label='Precision', marker='o', linestyle='-', color='blue')
plt.plot(epochs, history["val_recall"], label='Recall', marker='s', linestyle='-', color='red')
plt.title('Precision-Recall Trade-off Over Training Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('precision_recall.pdf')
plt.show()