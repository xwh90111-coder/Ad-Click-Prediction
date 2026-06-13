import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from sklearn.metrics import roc_auc_score, log_loss
import copy
import warnings
warnings.filterwarnings('ignore')

print("1. Loading processed data...")
data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
df = pd.read_csv(data_path)

if 'session_id' in df.columns:
    df.drop('session_id', axis=1, inplace=True)
if 'DateTime' in df.columns:
    df.drop('DateTime', axis=1, inplace=True)

target = ['is_click']
dense_features = ['city_development_index', 'var_1']
sparse_features = [col for col in df.columns if col not in dense_features + target]

print("2. Preprocessing features...")
df[sparse_features] = df[sparse_features].fillna('-1').astype(str)
df[dense_features] = df[dense_features].fillna(0)

for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

print("3. Configuring Regularized DeepFM architecture...")
# embedding_dim reduced to 4 to prevent over-memorization of IDs
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                           for feat in sparse_features] + \
                         [DenseFeat(feat, 1,) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

print("4. Splitting data into Train/Val/Test...")
# Use 80% train, 10% validation (for early stopping), 10% test (for final evaluation)
train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df[target])
train, val = train_test_split(train_val, test_size=0.1111, random_state=42, stratify=train_val[target])

train_model_input = {name: train[name] for name in feature_names}
val_model_input = {name: val[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Added Heavy L2 Regularization, shallower DNN, and high Dropout
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device,
               dnn_hidden_units=(128, 64), 
               l2_reg_linear=1e-3, 
               l2_reg_embedding=1e-3, 
               l2_reg_dnn=1e-3,
               dnn_dropout=0.3)

model.compile(optim.Adam(model.parameters(), lr=0.001), "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])

print("5. Training with Custom Early Stopping...")
epochs = 8
batch_size = 2048
best_val_auc = 0
best_weights = None
patience = 2
patience_counter = 0

for epoch in range(epochs):
    print(f"--- Epoch {epoch+1}/{epochs} ---")
    model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=1, verbose=2)
    
    # Evaluate on validation set
    val_pred = model.predict(val_model_input, batch_size=batch_size)
    val_auc = roc_auc_score(val[target].values, val_pred)
    val_loss = log_loss(val[target].values, val_pred)
    
    # Evaluate on train set to monitor overfitting gap
    train_pred = model.predict(train_model_input, batch_size=batch_size)
    train_auc = roc_auc_score(train[target].values, train_pred)
    
    print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_auc > best_val_auc:
        print(f"  >> Validation AUC improved from {best_val_auc:.4f} to {val_auc:.4f}. Saving weights...")
        best_val_auc = val_auc
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  >> No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early Stopping triggered!")
            break

print("\n6. Restoring Best Weights and Evaluating on Unseen Test Set...")
model.load_state_dict(best_weights)

pred_ans = model.predict(test_model_input, batch_size=batch_size)
final_auc = roc_auc_score(test[target].values, pred_ans)
final_loss = log_loss(test[target].values, pred_ans)

print("\n" + "="*40)
print("Anti-Overfitting DeepFM Final Evaluation:")
print(f"ROC AUC Score: {final_auc:.4f}")
print(f"Log Loss:      {final_loss:.4f}")
print("="*40)
