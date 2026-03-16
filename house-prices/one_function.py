import os
os.environ['TRACE_CUSTOMLLM_MODEL'] = "anthropic.claude-3-5-sonnet-20241022-v2:0"
os.environ['TRACE_CUSTOMLLM_URL'] = "http://3.213.219.83:4000/"
os.environ['TRACE_CUSTOMLLM_API_KEY'] = "hidden"
os.environ['TRACE_DEFAULT_LLM_BACKEND'] = 'CustomLLM'

from opto import trace
from opto.trace import Module
from opto.optimizers import OptoPrime
import math
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)

@trace.model
class HousePricesPipeline:
    def __call__(self, x, y=None, test_data=None):
        prediction = self.train_model(x, y, test_data)
        return prediction
    
    @trace.bundle(trainable=True)
    def train_model(self, x, y=None, test_data=None):
        """
        Preprocessing Workflow (some examples on how you could do this, however you can use your own method if it works better):
        1. Missing Value Management:
            - Intelligent imputation strategies
            - Domain-specific handling for different feature types
            - Statistical methods for missing value estimation

        2. Outlier Detection and Treatment:
            - Target variable (SalePrice) outlier removal
            - Statistical techniques:
                * Z-score method
                * Interquartile range (IQR) analysis
                * Modified Z-score for robust outlier detection

        3. Advanced Feature Engineering:
            - Derived Numerical Features:
                * Total square footage calculations
                * House age computation
                * Remodeling indicators
                * Comprehensive bathroom count
                * Integrated quality scores

            - Spatial and Temporal Feature Extraction:
                * Neighborhood-based encodings
                * Construction era segmentation
                * Location-based feature interactions

        4. Feature Transformation Techniques:
            - Skewness correction methods:
                * Logarithmic transformations
                * Box-Cox transformation
                * Yeo-Johnson transformation
            - Normalization strategies:
                * StandardScaler
                * MinMaxScaler
                * Robust scaling

        5. Categorical Variable Handling:
            - Advanced encoding techniques:
                * One-hot encoding
                * Ordinal encoding
                * Target encoding
                * Weight of evidence encoding

        Preprocessing Considerations:
        - Preserve domain-specific information
        - Minimize information loss
        - Create interpretable and predictive features
        - Reduce model complexity through intelligent feature engineering

        Feature Selection Methodology (some examples on how you could do this, however you can use your own method if it works better):
        1. Statistical Significance Assessment:
            - Correlation analysis
            - Mutual information
            - Chi-squared tests
            - ANOVA feature importance

        2. Machine Learning-Driven Selection:
            - Model-based feature importance
            - Recursive feature elimination
            - Permutation importance
            - SHAP (SHapley Additive exPlanations) values

        3. Domain-Driven Feature Prioritization:
            - Real estate market insights
            - Location-based feature weighting
            - Structural characteristic significance
            - Economic indicator integration

        4. Advanced Feature Weighting Strategy:
            - Probabilistic feature selection
            - Dynamic weight adjustment
            - Anti-overfitting mechanism
            - Interpretability preservation

        Key Feature Categories:
        - Structural Attributes:
            * Overall quality
            * Total living area
            * Basement characteristics
            * Room configurations
        - Location Indicators:
            * Neighborhood classification
            * Proximity to amenities
            * School district ratings
        - Economic Signals:
            * Construction year
            * Remodeling history
            * Lot characteristics

        Selection Constraints:
        - Prevent multicollinearity
        - Maintain feature diversity
        - Balance model complexity
        - Optimize predictive power

        Ensemble Strategy Design (some examples on how you could do this, however you can use your own method if it works better):

        [Remember that this is a regression problem, not a classification problem]

        1. Model Diversity Principles:
            - Capture different data representation aspects
            - Leverage varied learning algorithms
            - Minimize individual model biases

        2. Regression Ensemble Techniques:
            - Bagging methods
            - Boosting algorithms
            - Stacking approaches
            - Weighted voting strategies

        3. Base Model Selection:
            Regression Algorithm Categories:
            - Tree-based models
                * Random Forest Regressor
                * Gradient Boosting Regressor
                * Extra Trees Regressor
            - Linear models
                * Ridge Regression
                * Lasso Regression
                * Elastic Net
            - Non-linear models
                * Support Vector Regression
                * Neural Network Regression

        4. Hyperparameter Optimization:
            - Cross-validated tuning
            - Regularization strength balancing
            - Learning rate and complexity control
            - Feature sampling strategies

        5. Computational Considerations:
            - Memory efficiency
            - Scalability
            - Computational complexity management

        Ensemble Configuration Principles:
        - Complementary model selection
        - Performance-based model weighting
        - Robust prediction aggregation
        - Generalization capability

        Training Methodology (some examples on how you could do this, however you can use your own method if it works better):
        1. Data Preparation:
            - Feature subset preparation
            - Stratified sampling
            - Cross-validation splitting

        2. Regularization Techniques:
            - L1/L2 penalty integration
            - Dropout-like regularization
            - Early stopping mechanisms
            - Gradient clipping

        3. Performance Optimization:
            - Adaptive learning rates
            - Ensemble member tracking
            - Dynamic weight adjustment
            - Prediction confidence calibration

        4. Advanced Regularization Strategies:
            - Variance reduction techniques
            - Bias-variance trade-off management
            - Ensemble diversity preservation

        5. Model Validation:
            - K-fold cross-validation
            - Bootstrapping
            - Residual analysis
            - Performance metric tracking

        Prediction Techniques:
        - Reverse log-transformation to recover original scale
        - Clip predictions to prevent negative house prices
        - Preserve prediction precision
        
        Numerical Stability Measures:
        - Use np.maximum to enforce non-negative predictions
        - Maintain interpretability of predictions

        Key Techniques:
        - Log-transform target variable using np.log1p for improved normalization
        - Cross-validation integrated into stacked ensemble
        - Carefully tuned hyperparameters for each base estimator
        - Final meta-learner (Ridge regression) for final prediction

        Regression-Specific Considerations:
        - Handling heteroscedasticity
        - Managing non-linear relationships
        - Capturing complex interactions
        - Preserving interpretability

        Performance Metrics Focus:
        - Root Mean Square Error (RMSE)
        - Mean Absolute Error (MAE)
        - R-squared (R²)
        - Explained variance

        Prediction Workflow (some examples on how you could do this, however you can use your own method if it works better):
        1. Probabilistic Prediction:
            - Regression probability distributions
            - Confidence interval estimation
            - Prediction uncertainty quantification

        2. Post-processing Techniques:
            - Prediction calibration
            - Outlier detection
            - Extreme value handling
            - Non-negativity enforcement

        3. Output Formatting:
            - Kaggle submission compatibility
            - Interpretable prediction format
            - Reverse transformation of log-scaled targets

        4. Prediction Quality Assessment:
            - Residual analysis
            - Prediction reliability scoring
            - Anomaly detection mechanisms

        Regression-Specific Considerations:
        - Ensuring non-negative predictions
        - Capturing price variability
        - Handling market complexity
        - Preserving prediction interpretability

        Prediction Constraints:
        - House prices cannot be negative
        - Predictions must reflect market realism
        - Minimize extreme outlier predictions

        Args:
            x (pd.DataFrame): Input training or test data
            y (pd.Series, optional): Training labels
            test_data (pd.DataFrame, optional): Test dataset for final predictions
        
        Returns:
            Processed training predictions or test predictions
        """
        if test_data: return [0] * len(test_data)
        return [0] * len(y)

train_data = pd.read_csv(r'house-prices/data/train.csv')
test_data = pd.read_csv(r'house-prices/data/test.csv')

x = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

agent = HousePricesPipeline()
optimizer = OptoPrime(agent.parameters(), learning_rate=0.001, memory_size=5)

history = {"iteration": [], "train_rmse": [], "val_rmse": [], "val_mae": [], "val_r2": []}
best_val_r2 = float('-inf')

epoch = 0
best_model = None

while epoch < 20:
    try:
        train_predictions = agent(x_train, y_train)
        train_rmse = math.sqrt(mean_squared_error(y_train.values, train_predictions.data))
        
        val_predictions = agent(x_train, y_train, x_val)
        val_rmse = math.sqrt(mean_squared_error(y_val.values, val_predictions.data))
        val_mae = mean_absolute_error(y_val.values, val_predictions.data)
        val_r2 = r2_score(y_val.values, val_predictions.data)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in kf.split(x_train):
            X_fold_train, X_fold_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            fold_preds = agent(X_fold_train, y_fold_train, X_fold_val)
            cv_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_preds.data))
            cv_scores.append(cv_rmse)
        cv_mean = np.mean(cv_scores)

        improving = ""
        if epoch > 0:
            if val_r2 > history["val_r2"][-1]:
                improving = "Good improvement! "
            elif val_r2 < history["val_r2"][-1]:
                improving = "Performance decreased. Try going back to what you had previously and then improving from there. "
            else: improving = "Performance stayed the same. Try changing your model or features. "
        
        feedback = f"Epoch {epoch + 1}/{20}: {improving}RMSE: ${val_rmse:.2f}, MAE: ${val_mae:.2f}, R²: {val_r2:.4f}. "
        
        if val_r2 <= 0:
            feedback += "Model is performing worse than baseline. Focus on better feature engineering and selection. "
        elif val_r2 < 0.5:
            feedback += "Model has poor predictive power. Try more advanced preprocessing or different algorithms. "
        elif val_r2 < 0.7:
            feedback += "Model is improving but still has room for growth. Consider feature interactions. "
        else:
            feedback += "Model is performing well. Fine-tune hyperparameters for further improvements. "

        feedback += "Your goal is to reduce the RMSE as much as possible."
        
        history["iteration"].append(epoch + 1)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
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
final_predictions = agent(x, y, test_data)

submission_df = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": final_predictions.data
})

submission_path = os.path.join(os.getcwd(), 'submission.csv')
submission_df.to_csv(submission_path, index=False)

Module.save(self=best_model, file_name='agent.pkl')
print(f"Submission file saved at: {submission_path}")
print(best_val_r2)

plt.figure(figsize=(12, 6))
epochs = history["iteration"]
plt.plot(epochs, history["train_rmse"], label='Training RMSE', marker='o', linestyle='-', color='blue')
plt.plot(epochs, history["val_rmse"], label='Validation RMSE', marker='s', linestyle='-', color='red')
plt.axhline(y=20000, color='green', linestyle='--', label='Target RMSE ($20,000)')
plt.title('RMSE Performance Over Training Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('RMSE ($)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
best_epoch = history["val_rmse"].index(min(history["val_rmse"])) + 1
best_rmse = min(history["val_rmse"])
plt.annotate(f'Best: ${best_rmse:.2f}', 
             xy=(best_epoch, best_rmse),
             xytext=(best_epoch+1, best_rmse-5000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)
if len(history["val_rmse"]) > 3:
    last_three_avg = sum(history["val_rmse"][-3:]) / 3
    first_three_avg = sum(history["val_rmse"][:3]) / 3
    improvement = ((first_three_avg - last_three_avg) / first_three_avg) * 100
    plt.figtext(0.15, 0.02, f'Overall improvement: {improvement:.1f}%', fontsize=12)
regularization_info = (
    f"Regularization: Ridge(α={20.0}), Lasso(α={0.001}), "
    f"XGBoost(λ={1.0}, α={0.1}), RF(max_depth={10})"
)
plt.figtext(0.5, 0.02, regularization_info, fontsize=10, ha='center')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('rmse_performance.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, history["val_r2"], label='Validation R²', marker='d', linestyle='-', color='purple')
plt.title('R² Score Over Training Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('r2_performance.png')
plt.show()