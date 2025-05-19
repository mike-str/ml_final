import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
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

def load_and_clean_data():
    """Load and clean the data based on EDA findings."""
    print("Loading and cleaning data...")
    
    # Load the processed data from EDA
    df = pd.read_csv('processed_nba_data.csv')
    
    # Handle outliers in Inflation Adjusted Salary using IQR method
    Q1 = df['Inflation Adjusted Salary'].quantile(0.25)
    Q3 = df['Inflation Adjusted Salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove salary outliers
    df = df[(df['Inflation Adjusted Salary'] >= lower_bound) & (df['Inflation Adjusted Salary'] <= upper_bound)]
    print(f"Removed {(df['Inflation Adjusted Salary'] < lower_bound).sum() + (df['Inflation Adjusted Salary'] > upper_bound).sum()} salary outliers")
    
    return df

def create_features(df):
    """Create and select features based on VIF analysis and domain knowledge."""
    print("\nCreating and selecting features...")
    
    # Use the VIF-selected features from our EDA
    vif_selected_features = [
        'Games Started',
        '3-Pointers Made',
        '3-Point Percentage',
        'Free Throws Made',
        'Blocks per Game',
        'Rebounds per 36',
        'Assist to Turnover Ratio'
    ]
    
    # Add some additional features that might be important
    additional_features = [
        'Age',
        'Games',
        'Minutes Played',
        'Points per Game',
        'Assists per Game',
        'Steals per Game',
        'True Shooting %',
        'Usage Rate'
    ]
    
    # Combine all features
    selected_features = vif_selected_features + additional_features
    
    # Ensure all features exist in the dataframe
    available_features = [f for f in selected_features if f in df.columns]
    print(f"Using {len(available_features)} features for modeling")
    
    # Select features and target
    df_selected = df[available_features + ['Inflation Adjusted Salary']]
    
    # Drop rows with missing values
    before_drop = df_selected.shape[0]
    df_selected = df_selected.dropna()
    after_drop = df_selected.shape[0]
    dropped = before_drop - after_drop
    print(f"Dropped {dropped} rows with missing values. {after_drop} rows remain.")
    
    return df_selected

def standardize_features(df, target_col='Inflation Adjusted Salary'):
    """Standardize features and prepare train/test split."""
    print("\nStandardizing features...")
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    # Split the data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    # Save the scaler for later use
    import joblib
    joblib.dump(scaler, 'scaler.joblib')
    return X_train, X_test, y_train, y_test

def main():
    # Set up dual output (console and file)
    original_stdout = sys.stdout
    tee = TeeOutput('preprocess_output.txt')
    sys.stdout = tee
    try:
        # Load and clean data
        df = load_and_clean_data()
        # Create and select features
        df_processed = create_features(df)
        # Standardize features and create train/test split
        X_train, X_test, y_train, y_test = standardize_features(df_processed)
        # Save processed data
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        print("\nPreprocessing complete! Files saved:")
        print("- X_train.csv: Training features")
        print("- X_test.csv: Test features")
        print("- y_train.csv: Training target (Inflation Adjusted Salary)")
        print("- y_test.csv: Test target (Inflation Adjusted Salary)")
        print("- scaler.joblib: StandardScaler for future use")
    finally:
        sys.stdout = original_stdout
        tee.close()

if __name__ == "__main__":
    main() 