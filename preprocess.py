import pandas as pd
import os

train_path = r"D:\projects\Ad Click Pridiction\archive\Ad_click_prediction_train (1).csv"
test_path = r"D:\projects\Ad Click Pridiction\archive\Ad_Click_prediciton_test.csv"
out_train_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
out_test_path = r"D:\projects\Ad Click Pridiction\archive\processed_test.csv"

def process_data(df):
    df = df.copy()
    
    print("  Extracting time features...")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['day'] = df['DateTime'].dt.day
    
    missing_stats = df.isnull().sum()
    print(f"  Missing values before processing:\n{missing_stats[missing_stats > 0]}")

    categorical_cols_with_nan = ['product_category_2', 'user_group_id', 'age_level', 'user_depth']
    for col in categorical_cols_with_nan:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('Unknown')
        
    if 'city_development_index' in df.columns:
        median_val = df['city_development_index'].median()
        df['city_development_index'] = df['city_development_index'].fillna(median_val)
        
    return df

print("Loading training data...")
train = pd.read_csv(train_path)
print("Loading test data...")
test = pd.read_csv(test_path)

print("\n--- Processing Train Data ---")
train_processed = process_data(train)

print("\n--- Processing Test Data ---")
test_processed = process_data(test)

print("\nSaving processed data to archive folder...")
train_processed.to_csv(out_train_path, index=False)
test_processed.to_csv(out_test_path, index=False)
print(f"Done! Files saved at:\n{out_train_path}\n{out_test_path}")