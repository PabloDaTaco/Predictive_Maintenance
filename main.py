import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("ai4i2020.csv")

# Convert product Type into numerical columns
data = pd.get_dummies(data, columns=["Type"], dtype=int)

# Chosen Features (inputs)
X = data[[
    "Type_L",
    "Type_M",
    "Type_H",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

# Target (what we want to predict)
y = data["Machine failure"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,     # 20% of the rows into testing
    random_state=42,    # 42 is just a random chosen row
    stratify=y          # Pays attention to 0s and 1s in Y when splitting
)

print("X:", X.shape)
print("y:", y.shape)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)