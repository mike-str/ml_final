import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import shutil
import os
import time
from datetime import datetime
import sys
from contextlib import contextmanager
warnings.filterwarnings('ignore')

class TeeOutput:
    """Class to handle writing to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8', errors='replace')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def print_section(title):
    """Print a formatted section title."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")

def load_data():
    """Load the preprocessed data and apply log transformation to target variable."""
    print_section("Loading Data")
    print("Loading preprocessed data...")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    
    # Apply log transformation to target variable
    y_train_log = np.log1p(y_train)  # log1p for handling zeros if any
    y_test_log = np.log1p(y_test)
    
    print(f"✓ Data loaded successfully")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Target variable (Inflation Adjusted Salary) transformed using log1p")
    
    return X_train, X_test, y_train, y_test, y_train_log, y_test_log

def evaluate_model_with_cv(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    print_section(f"Evaluating {model_name}")
    start_time = time.time()
    # 4-fold cross-validation with grid search
    print("Performing 4-fold cross-validation with grid search for hyperparameter tuning...")
    grid = GridSearchCV(model, param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid.fit(X_train, np.log1p(y_train))
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(f"Best hyperparameters: {best_params}")
    # Cross-validation RMSE (log scale)
    cv_rmse = np.sqrt(-grid.best_score_)
    print(f"Best CV RMSE (log scale): {cv_rmse:.4f}")
    # Retrain best model on full training set
    best_model.fit(X_train, np.log1p(y_train))
    # Predictions
    y_pred_train_log = best_model.predict(X_train)
    y_pred_test_log = best_model.predict(X_test)
    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_test = np.expm1(y_pred_test_log)
    # Metrics
    metrics = {
        'Model': model_name,
        'Best Params': best_params,
        'CV RMSE (log scale)': cv_rmse,
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train R2': r2_score(y_train, y_pred_train),
        'Test R2': r2_score(y_test, y_pred_test),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Test MAE': mean_absolute_error(y_test, y_pred_test)
    }
    print("\nKey Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:,.4f}" if 'RMSE' in k or 'MAE' in k else f"  - {k}: {v}")
        else:
            print(f"  - {k}: {v}")
    end_time = time.time()
    print(f"\n✓ Model evaluation completed in {end_time - start_time:.2f} seconds")
    return metrics, best_model

def plot_predictions(y_true, y_pred, model_name, set_name):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Inflation Adjusted Salary ($)')
    plt.ylabel('Predicted Inflation Adjusted Salary ($)')
    plt.title(f'{model_name} - {set_name} Set: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'model_img/{model_name.lower().replace(" ", "_")}_{set_name.lower()}_predictions.png')
    plt.close()

def plot_residuals(y_true, y_pred, model_name, set_name):
    """Plot residuals analysis."""
    residuals = y_true - y_pred
    
    # Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Inflation Adjusted Salary ($)')
    plt.ylabel('Residuals ($)')
    plt.title(f'{model_name} - {set_name} Set: Residuals vs Predicted')
    plt.tight_layout()
    plt.savefig(f'model_img/{model_name.lower().replace(" ", "_")}_{set_name.lower()}_residuals.png')
    plt.close()
    
    # Residuals Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals ($)')
    plt.ylabel('Count')
    plt.title(f'{model_name} - {set_name} Set: Residuals Distribution')
    plt.tight_layout()
    plt.savefig(f'model_img/{model_name.lower().replace(" ", "_")}_{set_name.lower()}_residuals_dist.png')
    plt.close()

def plot_learning_curves(model, X_train, y_train_log, model_name):
    """Plot learning curves to analyze model's learning process."""
    print(f"\nGenerating learning curves for {model_name}...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train_log, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_scores = np.sqrt(-train_scores)
    test_scores = np.sqrt(-test_scores)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE (log scale)')
    plt.title(f'{model_name} - Learning Curves')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'model_img/{model_name.lower().replace(" ", "_")}_learning_curves.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'{model_name} - Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'model_img/{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()

def plot_metrics_comparison(results_df):
    """Plot comparison of different metrics across models."""
    print_section("Generating Model Comparison Plots")
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Model', y='Test RMSE')
    plt.title('Model Comparison - Test RMSE (Inflation Adjusted)')
    plt.ylabel('RMSE ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_img/model_comparison_rmse.png')
    plt.close()
    
    # Plot R² comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Model', y='Test R2')
    plt.title('Model Comparison - Test R²')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_img/model_comparison_r2.png')
    plt.close()
    
    # Plot MAE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Model', y='Test MAE')
    plt.title('Model Comparison - Test MAE (Inflation Adjusted)')
    plt.ylabel('MAE ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_img/model_comparison_mae.png')
    plt.close()

def main():
    # Set up dual output (console and file)
    original_stdout = sys.stdout
    tee = TeeOutput('train_models_output.txt')
    sys.stdout = tee
    try:
        print_section("Starting Model Training and Evaluation")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if os.path.exists('model_img'):
            shutil.rmtree('model_img')
        os.makedirs('model_img')
        print("✓ Created fresh model_img directory")
        # Load data
        X_train, X_test, y_train, y_test, _, _ = load_data()
        # Define models and hyperparameter grids
        models_and_grids = [
            (LinearRegression(), {}, 'Linear Regression'),
            (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}, 'Random Forest'),
            (XGBRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [3, 6]}, 'XGBoost'),
            (SVR(), {'C': [1, 10], 'gamma': ['scale', 'auto']}, 'SVR')
        ]
        results = []
        best_model = None
        best_test_rmse = float('inf')
        for model, param_grid, model_name in models_and_grids:
            metrics, trained_model = evaluate_model_with_cv(
                model, param_grid, X_train, y_train, X_test, y_test, model_name
            )
            results.append(metrics)
            if metrics['Test RMSE'] < best_test_rmse:
                best_test_rmse = metrics['Test RMSE']
                best_model = trained_model
                print(f"\n✓ New best model: {model_name} (Test RMSE: ${best_test_rmse:,.2f})")
        # Save results
        results_df = pd.DataFrame(results)
        print_section("Model Comparison Results")
        print(results_df.to_string(index=False))
        results_df.to_csv('model_comparison_results.csv', index=False)
        print("\n✓ Saved detailed results to model_comparison_results.csv")
        joblib.dump(best_model, 'best_model.joblib')
        print(f"\n✓ Best model saved as 'best_model.joblib'")
        plot_metrics_comparison(results_df)
        print_section("Analysis Completed")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        sys.stdout = original_stdout
        tee.close()

if __name__ == "__main__":
    main() 