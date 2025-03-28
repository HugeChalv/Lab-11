{python}
############################
# ALL-IN-ONE CODE SNIPPET
############################

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader

###################################
# 1. LOAD AND PREPARE THE DATASET
###################################

# Replace with your own CSV path
data_path = 'lab_11_bridge_data.csv'
df = pd.read_csv(data_path)

print("Data head:\n", df.head())
print("\nData info:\n", df.info())

# Suppose 'Bridge_ID' is just an ID column (non-informative), drop it
if 'Bridge_ID' in df.columns:
    df.drop(columns=['Bridge_ID'], inplace=True)

# Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Example: drop rows with any missing value
df.dropna(inplace=True)

# Separate features and target
target_col = 'Max_Load_Tons'
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numeric & categorical features
numeric_features = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Condition_Rating']
categorical_features = ['Material']

# Build a ColumnTransformer for numeric scaling + categorical one-hot
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))  # drop='first' to avoid dummy trap
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit the preprocessor
preprocessor.fit(X)

# Transform X
X_processed = preprocessor.transform(X)

# Convert to float32 numpy arrays
X_processed = X_processed.astype(np.float32)
y = y.values.astype(np.float32).reshape(-1, 1)

#############################################
# 2. SPLIT DATA INTO TRAIN AND VALIDATION
#############################################

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

###################################
# 3. PYTORCH DATASET AND DATALOADER
###################################

class BridgeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create the Datasets
train_dataset = BridgeDataset(X_train, y_train)
val_dataset   = BridgeDataset(X_val, y_val)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

###################################
# 4. DEFINE THE MODEL
###################################

# Get number of input features
input_dim = X_train.shape[1]

class BridgeLoadModel(nn.Module):
    def __init__(self, input_dim):
        super(BridgeLoadModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = BridgeLoadModel(input_dim)

# Loss & Optimizer (with L2 reg)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

###################################
# 5. TRAINING LOOP WITH EARLY STOP
###################################

num_epochs = 100
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    #################
    # TRAINING
    #################
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        # Forward
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    #################
    # VALIDATION
    #################
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

# Load the best model weights
model.load_state_dict(best_model_weights)

###################################
# 6. OPTIONAL: SAVE MODEL/PREPROCESSOR
###################################
# import joblib
# joblib.dump(preprocessor, 'preprocessor.pkl')
# torch.save(model.state_dict(), 'bridge_load_model.pth')

print("\nTraining complete. Best validation loss:", best_val_loss)
