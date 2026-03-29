import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Customer-Churn-Records.csv")

# Drop unnecessary columns
data = data.drop(columns=['CustomerId','RowNumber','Surname','Complain'])

# Convert categorical to numeric
data = pd.get_dummies(data)

# Split
X = data.drop("Exited", axis=1)
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model saved as model.pkl")