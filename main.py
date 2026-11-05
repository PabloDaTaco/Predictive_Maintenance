import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt
import graphviz

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

#print("X:", X.shape)
#print("y:", y.shape)

#print("X_train:", X_train.shape)
#print("y_train:", y_train.shape)

#print("X_test:", X_test.shape)
#print("y_test:", y_test.shape)





dt_model = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\nDECISION TREE RESULTS")
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))

# After analyzing the results and confusion matrix, precision score is 77% 
# meaning the model is correct 77% of the time when predicting if a machine will fail.
# The recall score is 29% which means out of all the machines that actually failed, 
# our model only identified about 29%.
# The F1-score came out low since its a balance of both precision and recall at about 43%.

# The confusion matrix tells us that in the 2,000 testing observations, 1,932 of those
# observations were True Negatives, 6 were False Positives, 48 were False Negative, and 
# 20 were True Positives. Although our model achieved approximately 97% accuracy, the 
# confusion matrix reveals that it missed most of the actual machine failures.


# After adding balanced class weights, the model became much better at identifying 
# actual machine failures. Previously, it only detected 20 out of 68 failures. After balancing, 
# it detected 60 out of 68 failures. This increased recall from 29% to 88%.


# Adding balanced class weights helped address the class imbalance in the dataset. 
# As a result, the model became much better at detecting machine failures. Recall for 
# the failure class improved from 29% to 88%, meaning the model now catches most failures. 
# However, this also increased false positives, so the model produces more false alarms. 
# For predictive maintenance, this tradeoff may be acceptable because missing real failures 
# is often more costly than unnecessary maintenance checks.



rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))





# Create Logistic Regression model
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test)

# Evaluate performance
print("\nLOGISTIC REGRESSION RESULTS")
print(classification_report(y_test, lr_pred, zero_division=0))
print(confusion_matrix(y_test, lr_pred))

#Plot graphs of Decision Tree and confusion matrix
#Decision tree output is a little rough will fix
plt.figure(figsize=(20, 10))

dot_data = export_graphviz(
    dt_model,
    out_file=None,
    feature_names=X.columns,
    class_names=["No Failure", "Failure"],
    filled=True,
    rounded=True,
)

graph = graphviz.Source(dot_data)

graph.render("Decision_Tree", format="png", cleanup=True)
graph.view()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

models = [
    ("Decision Tree", dt_pred),
    ("Random Forest", rf_pred),
    ("Logistic Regression", lr_pred)
]

for ax, (name, prediction) in zip(axes, models):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        prediction,
        display_labels=["No Failure", "Failure"],
        cmap="Blues",
        ax=ax,
        colorbar=False
    )
    ax.set_title(name)


plt.tight_layout()
plt.show()

# Select numerical features for correlation analysis
eda_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

# Calculate correlations
correlation = data[eda_features].corr()

# Generate heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Machine Operating Features")
plt.tight_layout()


plt.show()


# Create scatter plot
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=data,
    x="Rotational speed [rpm]",
    y="Torque [Nm]",
    hue="Machine failure",
    palette={0: "steelblue", 1: "red"},
    alpha=0.5
)

plt.title("Rotational Speed vs. Torque by Machine Failure")
plt.xlabel("Rotational Speed (rpm)")
plt.ylabel("Torque (Nm)")

# Label the classes
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles,
    ["No Failure", "Failure"],
    title="Machine Status"
)

plt.tight_layout()
plt.show()
