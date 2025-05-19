import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
import shutil
from datetime import datetime
import sys

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

# ========== SETUP AND CONFIGURATION ========== #
print("Starting comprehensive NBA salary analysis...")

# Delete existing plot_img directory if it exists
if os.path.exists('plot_img'):
    shutil.rmtree('plot_img')
    print("Cleaned up existing plot_img directory")

# Create plot directories with clear structure
plot_dirs = {
    '1_data_quality': 'plot_img/1_data_quality',
    '2_univariate': 'plot_img/2_univariate',
    '3_bivariate': 'plot_img/3_bivariate',
    '4_multivariate': 'plot_img/4_multivariate',
    '5_feature_importance': 'plot_img/5_feature_importance',
    '6_salary_analysis': 'plot_img/6_salary_analysis',
    '7_advanced_metrics': 'plot_img/7_advanced_metrics'
}

for dir_path in plot_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ========== DATA LOADING AND INITIAL PROCESSING ========== #
print("\n1. Loading and preprocessing data...")

# Load data
df = pd.read_csv('merged_player_data.csv')

# Rename columns for clarity
df = df.rename(columns={
    "Pos": "Position",
    "Tm": "Team",
    "G": "Games",
    "GS": "Games Started",
    "MP": "Minutes Played",
    "FG": "Field Goals Made",
    "FGA": "Field Goals Attempted",
    "FG%": "Field Goal Percentage",
    "3P": "3-Pointers Made",
    "3PA": "3-Pointers Attempted",
    "3P%": "3-Point Percentage",
    "2P": "2-Pointers Made",
    "2PA": "2-Pointers Attempted",
    "2P%": "2-Point Percentage",
    "eFG%": "Effective Field Goal Percentage",
    "FT": "Free Throws Made",
    "FTA": "Free Throws Attempted",
    "FT%": "Free Throw Percentage",
    "ORB": "Offensive Rebounds",
    "DRB": "Defensive Rebounds",
    "TRB": "Total Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
    "PF": "Personal Fouls",
    "PTS": "Points",
    "Salary": "Base Salary",
    "InflationAdjSalary": "Inflation Adjusted Salary"
})

# Convert numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ========== 1. DATA QUALITY ANALYSIS ========== #
print("\n2. Performing data quality analysis...")

# 1.1 Missing Value Analysis
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_analysis = pd.DataFrame({
    'Missing Values': missing_data,
    'Percentage': missing_percentage
}).sort_values('Percentage', ascending=False)

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('1.1 Missing Data Pattern Analysis')
plt.savefig(f'{plot_dirs["1_data_quality"]}/1.1_missing_data_pattern.png')
plt.close()

# Print missing data summary
if missing_data.sum() == 0:
    print("No missing data detected in the dataset.")
else:
    print("Missing data detected! See 1.1_missing_data_pattern.png and 1.3_outlier_analysis.csv for details.")
    print(missing_analysis[missing_analysis['Missing Values'] > 0])

# 1.2 Data Distribution Overview
plt.figure(figsize=(15, 8))
df['Base Salary'].hist(bins=50)
plt.title('1.2 Base Salary Distribution')
plt.xlabel('Base Salary')
plt.ylabel('Frequency')
plt.savefig(f'{plot_dirs["1_data_quality"]}/1.2_salary_distribution.png')
plt.close()

# 1.3 Outlier Detection
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_analysis = pd.DataFrame()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
    outlier_analysis.loc[col, 'Outlier Count'] = len(outliers)
    outlier_analysis.loc[col, 'Outlier Percentage'] = (len(outliers) / len(df)) * 100

outlier_analysis = outlier_analysis.sort_values('Outlier Percentage', ascending=False)
outlier_analysis.to_csv(f'{plot_dirs["1_data_quality"]}/1.3_outlier_analysis.csv')

# Print outlier summary
outlier_cols = outlier_analysis[outlier_analysis['Outlier Count'] > 0]
if outlier_cols.empty:
    print("No significant outliers detected in numeric features.")
else:
    print("Outliers detected in the following features:")
    print(outlier_cols[['Outlier Count', 'Outlier Percentage']])

# ========== 2. UNIVARIATE ANALYSIS ========== #
print("\n3. Performing univariate analysis...")

# 2.1 Salary Distribution by Position
plt.figure(figsize=(12, 6))
sns.boxplot(x='Position', y='Base Salary', data=df)
plt.title('2.1 Salary Distribution by Position')
plt.xticks(rotation=45)
plt.savefig(f'{plot_dirs["2_univariate"]}/2.1_salary_by_position.png')
plt.close()

# 2.2 Age Distribution and Salary
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Base Salary', data=df, alpha=0.5)
plt.title('2.2 Age vs Salary Distribution')
plt.savefig(f'{plot_dirs["2_univariate"]}/2.2_age_vs_salary.png')
plt.close()

# 2.3 Performance Metrics Distribution
performance_metrics = ['Points', 'Assists', 'Total Rebounds', 'Field Goal Percentage']
for i, metric in enumerate(performance_metrics, 1):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=metric, bins=30)
    plt.title(f'2.3.{i} Distribution of {metric}')
    plt.savefig(f'{plot_dirs["2_univariate"]}/2.3.{i}_{metric}_distribution.png')
    plt.close()

# ========== 3. BIVARIATE ANALYSIS ========== #
print("\n4. Performing bivariate analysis...")

# 3.1 Correlation Analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('3.1 Correlation Matrix of Numeric Features')
plt.savefig(f'{plot_dirs["3_bivariate"]}/3.1_correlation_matrix.png')
plt.close()

# 3.2 Salary vs Key Performance Indicators
key_metrics = ['Points', 'Assists', 'Total Rebounds', 'Field Goal Percentage']
for i, metric in enumerate(key_metrics, 1):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=metric, y='Base Salary', data=df, scatter_kws={'alpha':0.3})
    plt.title(f'3.2.{i} {metric} vs Salary')
    plt.savefig(f'{plot_dirs["3_bivariate"]}/3.2.{i}_{metric}_vs_salary.png')
    plt.close()

# ========== 4. MULTIVARIATE ANALYSIS ========== #
print("\n5. Performing multivariate analysis...")

# 4.1 Position and Age Interaction
df['Age Group'] = pd.cut(df['Age'], bins=[0, 24, 28, 32, 100], labels=['<=24', '25-28', '29-32', '33+'])
plt.figure(figsize=(12, 8))
sns.boxplot(x='Position', y='Base Salary', hue='Age Group', data=df)
plt.title('4.1 Salary by Position and Age Group')
plt.xticks(rotation=45)
plt.savefig(f'{plot_dirs["4_multivariate"]}/4.1_salary_position_age.png')
plt.close()

# 4.2 Performance Metrics Interaction
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Points', y='Assists', hue='Base Salary', size='Base Salary',
                sizes=(20, 200), data=df)
plt.title('4.2 Points vs Assists Colored by Salary')
plt.savefig(f'{plot_dirs["4_multivariate"]}/4.2_points_assists_salary.png')
plt.close()

# ========== 5. FEATURE IMPORTANCE ANALYSIS ========== #
print("\n6. Analyzing feature importance...")

# 5.1 Add Derived Features
print("\nAdding derived features...")

# Calculate shooting percentages (with division by zero handling)
df['Field Goal Percentage'] = (df['Field Goals Made'] / df['Field Goals Attempted'].replace(0, np.nan)) * 100
df['2-Point Percentage'] = (df['2-Pointers Made'] / df['2-Pointers Attempted'].replace(0, np.nan)) * 100
df['3-Point Percentage'] = (df['3-Pointers Made'] / df['3-Pointers Attempted'].replace(0, np.nan)) * 100
df['Free Throw Percentage'] = (df['Free Throws Made'] / df['Free Throws Attempted'].replace(0, np.nan)) * 100

# Calculate per game stats (with division by zero handling)
df['Points per Game'] = df['Points'] / df['Games'].replace(0, np.nan)
df['Assists per Game'] = df['Assists'] / df['Games'].replace(0, np.nan)
df['Rebounds per Game'] = df['Total Rebounds'] / df['Games'].replace(0, np.nan)
df['Steals per Game'] = df['Steals'] / df['Games'].replace(0, np.nan)
df['Blocks per Game'] = df['Blocks'] / df['Games'].replace(0, np.nan)
df['Turnovers per Game'] = df['Turnovers'] / df['Games'].replace(0, np.nan)

# Calculate per 36 minutes stats (with division by zero handling)
df['Points per 36'] = (df['Points'] / df['Minutes Played'].replace(0, np.nan)) * 36
df['Assists per 36'] = (df['Assists'] / df['Minutes Played'].replace(0, np.nan)) * 36
df['Rebounds per 36'] = (df['Total Rebounds'] / df['Minutes Played'].replace(0, np.nan)) * 36

# Calculate advanced metrics (with division by zero handling)
df['True Shooting %'] = df['Points'] / (2 * (df['Field Goals Attempted'] + 0.44 * df['Free Throws Attempted']).replace(0, np.nan)) * 100
df['Usage Rate'] = (df['Field Goals Attempted'] + 0.44 * df['Free Throws Attempted'] + df['Turnovers']) / df['Games'].replace(0, np.nan)
df['Assist to Turnover Ratio'] = df['Assists'] / df['Turnovers'].replace(0, np.nan)

# Replace inf and -inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# 5.2 Iterative VIF Analysis
print("\nPerforming VIF analysis to handle multicollinearity...")

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

# Prepare data for VIF analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
temp = df[numeric_cols].drop(['Base Salary', 'Inflation Adjusted Salary'], axis=1, errors='ignore')
temp = temp.fillna(temp.mean())  # Handle missing values

# Store initial VIF values
initial_vif = calculate_vif(temp)
initial_vif.to_csv(f'{plot_dirs["5_feature_importance"]}/5.1_initial_vif_values.csv')

# Plot initial VIF values
plt.figure(figsize=(12, 8))
sns.barplot(x='VIF', y='Feature', data=initial_vif.sort_values('VIF', ascending=False).head(15))
plt.title('5.1 Initial VIF Values (Top 15)')
plt.savefig(f'{plot_dirs["5_feature_importance"]}/5.1_initial_vif_values.png')
plt.close()

# Iterative VIF removal
removed_features = []
while True:
    vif = calculate_vif(temp)
    max_vif = vif["VIF"].max()
    if max_vif > 5:
        drop_feature = vif.sort_values("VIF", ascending=False)["Feature"].iloc[0]
        removed_features.append((drop_feature, max_vif))
        temp = temp.drop(columns=[drop_feature])
    else:
        break

# Save removed features and their VIF values
removed_features_df = pd.DataFrame(removed_features, columns=['Feature', 'VIF'])
removed_features_df.to_csv(f'{plot_dirs["5_feature_importance"]}/5.2_removed_features_vif.csv')

# Print removed features in a more organized way
print("\nFeatures removed due to high VIF (> 5):")
for feature, vif_value in removed_features:
    print(f"- {feature}: VIF = {vif_value:.2f}")

print(f"\nSelected {len(temp.columns)} features after VIF analysis:")
print("- " + "\n- ".join(temp.columns))

# Plot final VIF values
final_vif = calculate_vif(temp)
plt.figure(figsize=(12, 8))
sns.barplot(x='VIF', y='Feature', data=final_vif.sort_values('VIF', ascending=False))
plt.title('5.2 Final VIF Values After Feature Removal')
plt.savefig(f'{plot_dirs["5_feature_importance"]}/5.2_final_vif_values.png')
plt.close()

# 5.3 Random Forest Feature Importance (using VIF-selected features)
X = temp  # Use the features after VIF selection
y = df['Base Salary']

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('5.3 Top 15 Features for Salary Prediction (After VIF Selection)')
plt.savefig(f'{plot_dirs["5_feature_importance"]}/5.3_feature_importance_vif_selected.png')
plt.close()

# Save the VIF-selected features for later use
vif_selected_features = pd.DataFrame({'Feature': temp.columns})
vif_selected_features.to_csv(f'{plot_dirs["5_feature_importance"]}/5.4_vif_selected_features.csv', index=False)

# ========== 6. SALARY-SPECIFIC ANALYSIS ========== #
print("\n7. Performing salary-specific analysis...")

# 6.1 Salary Percentiles by Position
salary_percentiles = df.groupby('Position')['Base Salary'].agg(['mean', 'median', 'std']).round(2)
salary_percentiles.to_csv(f'{plot_dirs["6_salary_analysis"]}/6.1_salary_percentiles.csv')

plt.figure(figsize=(12, 6))
sns.boxplot(x='Position', y='Base Salary', data=df)
plt.title('6.1 Salary Distribution by Position')
plt.xticks(rotation=45)
plt.savefig(f'{plot_dirs["6_salary_analysis"]}/6.1_salary_by_position.png')
plt.close()

# 6.2 Salary vs Experience
if 'Experience' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Experience', y='Base Salary', data=df, alpha=0.5)
    plt.title('6.2 Salary vs Experience')
    plt.savefig(f'{plot_dirs["6_salary_analysis"]}/6.2_salary_vs_experience.png')
    plt.close()

# ========== 7. ADVANCED METRICS ========== #
print("\n8. Calculating advanced metrics...")

# 7.1 Efficiency Metrics
df['Points per Game'] = df['Points'] / df['Games']
df['Assists per Game'] = df['Assists'] / df['Games']
df['Rebounds per Game'] = df['Total Rebounds'] / df['Games']

# 7.2 Advanced Shooting Metrics
df['True Shooting %'] = df['Points'] / (2 * (df['Field Goals Attempted'] + 0.44 * df['Free Throws Attempted']))
df['Usage Rate'] = (df['Field Goals Attempted'] + 0.44 * df['Free Throws Attempted'] + df['Turnovers']) / df['Games']

# Plot advanced metrics vs salary
advanced_metrics = ['Points per Game', 'Assists per Game', 'Rebounds per Game', 'True Shooting %']
for i, metric in enumerate(advanced_metrics, 1):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=metric, y='Base Salary', data=df, alpha=0.5)
    plt.title(f'7.{i} {metric} vs Salary')
    plt.savefig(f'{plot_dirs["7_advanced_metrics"]}/7.{i}_{metric}_vs_salary.png')
    plt.close()

# Save processed data
df.to_csv('processed_nba_data.csv', index=False)

print("\nAnalysis complete! All plots and data have been saved in their respective directories.")
print("Key findings have been saved in the plot directories for reference.")

def plot_salary_distributions(df):
    """Plot salary distributions: original (linear x-axis) and log-transformed (linear x-axis)."""
    salary_col = 'Inflation Adjusted Salary' if 'Inflation Adjusted Salary' in df.columns else 'InflationAdjSalary'
    salaries = df[salary_col].dropna()
    log_salaries = np.log1p(salaries)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Original salary with linear x-axis
    sns.histplot(salaries, bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_xlabel('Inflation Adjusted Salary ($)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Original Salary Distribution')
    # Log-transformed salary
    sns.histplot(log_salaries, bins=50, kde=True, ax=axes[1], color='salmon')
    axes[1].set_xlabel('Log(1 + Inflation Adjusted Salary)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Log-Transformed Salary Distribution')
    plt.tight_layout()
    plt.savefig('eda_img/salary_distributions.png')
    plt.close()

def plot_feature_vs_salary(df, feature):
    """Plot feature vs salary relationship."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature, y='InflationAdjSalary', alpha=0.5)
    plt.title(f'{feature} vs Inflation Adjusted Salary')
    plt.tight_layout()
    plt.savefig(f'eda_img/{feature}_vs_salary.png')
    plt.close()

def plot_categorical_vs_salary(df, feature):
    """Plot categorical feature vs salary relationship."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=feature, y='InflationAdjSalary')
    plt.title(f'{feature} vs Inflation Adjusted Salary')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'eda_img/{feature}_vs_salary.png')
    plt.close()

def plot_correlation_matrix(df):
    """Plot correlation matrix of numerical features."""
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('eda_img/correlation_matrix.png')
    plt.close()
    
    # Print top correlations with salary
    salary_correlations = corr_matrix['InflationAdjSalary'].sort_values(ascending=False)
    print("\nTop correlations with Inflation Adjusted Salary:")
    print(salary_correlations.head(10))

def plot_feature_distribution(df, feature):
    """Plot histogram and boxplot for a numerical feature."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature].dropna(), bins=30, kde=True)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[feature].dropna(), orient='h')
    plt.title(f'{feature} Boxplot')
    plt.xlabel(feature)
    
    plt.tight_layout()
    plt.savefig(f'eda_img/{feature}_distribution.png')
    plt.close()

def plot_categorical_distribution(df, feature):
    """Plot countplot for a categorical feature."""
    plt.figure(figsize=(12, 6))
    order = df[feature].value_counts().index
    sns.countplot(data=df, x=feature, order=order)
    plt.title(f'{feature} Countplot')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'eda_img/{feature}_countplot.png')
    plt.close()

def main():
    # Set up dual output (console and file)
    original_stdout = sys.stdout
    tee = TeeOutput('eda_output.txt')
    sys.stdout = tee
    try:
        print(f"\nStarting EDA at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create and clean eda_img directory
        if os.path.exists('eda_img'):
            shutil.rmtree('eda_img')
        os.makedirs('eda_img')
        print("Created fresh eda_img directory")
        
        # Load data
        df = pd.read_csv('merged_player_data.csv')
        
        # Basic data overview
        print("\nBasic Data Overview:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumns in the dataset:")
        for col in df.columns:
            print(f"- {col}")
        
        # Check for missing values
        print("\nMissing Values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Plot salary distributions
        plot_salary_distributions(df)
        
        # Analyze numerical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        print("\nNumerical Features Analysis:")
        for feature in numerical_features:
            if feature not in ['Inflation Adjusted Salary', 'Base Salary']:  # Skip salary columns
                print(f"\nAnalyzing {feature}...")
                plot_feature_distribution(df, feature)
                plot_feature_vs_salary(df, feature)
        
        # Analyze categorical features
        categorical_features = df.select_dtypes(include=['object']).columns
        print("\nCategorical Features Analysis:")
        for feature in categorical_features:
            print(f"\nAnalyzing {feature}...")
            plot_categorical_distribution(df, feature)
            plot_categorical_vs_salary(df, feature)
        
        # Correlation analysis
        print("\nAnalyzing correlations...")
        plot_correlation_matrix(df)
        
        print(f"\nEDA completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        sys.stdout = original_stdout
        tee.close()

if __name__ == "__main__":
    main() 