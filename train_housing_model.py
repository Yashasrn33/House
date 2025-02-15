import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import joblib

# Load the dataset
data = pd.read_csv("Housing.csv")

# Preprocess the dataset (Handle missing data, encode categorical columns)
# One-hot encoding for categorical columns
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Fill missing values with median (in case there are any)
data.fillna(data.median(), inplace=True)

# Split into features and target
X = data.drop('price', axis=1)
y = data['price']

# Save feature names for API validation
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "feature_scaler.pkl")  # Save feature scaler

# Normalize the target variable (price)
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))
joblib.dump(target_scaler, "target_scaler.pkl")  # Save target scaler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Dataset Class
class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader for training and test sets
train_dataset = HousingDataset(X_train_tensor, y_train_tensor)
test_dataset = HousingDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
input_size = X_train.shape[1]
model = RegressionModel(input_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

    # Inverse transform the predictions to the original scale
    predictions_original = target_scaler.inverse_transform(predictions.detach().numpy())
    print("Predictions (original scale):", predictions_original[:10])  # Display first 10 predictions

from sklearn.metrics import mean_absolute_error, r2_score

# Get predictions for the test data
predictions = model(X_test_tensor)
predictions_original = target_scaler.inverse_transform(predictions.detach().numpy())

# Calculate MAE and R² score
y_test_original = target_scaler.inverse_transform(y_test_tensor.detach().numpy())
mae = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'R² Score: {r2}')

# Save the trained model
torch.save(model.state_dict(), "house_price_model.pth")
