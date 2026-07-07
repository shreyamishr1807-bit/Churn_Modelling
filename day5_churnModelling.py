import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Check missing values
print("\nMissing Values:")
print(dataset.isnull().sum())
print(dataset.head())
dataset.dropna(inplace=True)
print(dataset.shape)
# Remove unnecessary columns
dataset = dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
dataset["Gender"] = le_gender.fit_transform(dataset["Gender"])
# One Hot Encoding the "Geography" column
dataset=pd.get_dummies(dataset,columns=["Geography"],drop_first=True,dtype=int)
X = dataset.drop("Exited", axis=1)
y = dataset["Exited"]
dataset.head()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42) 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))

# Hidden Layer 2
model.add(Dense(units=8, activation='relu'))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
loss, accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Predict on test data
y_pred = model.predict(X_test)

# Convert probabilities to 0 or 1
y_pred = (y_pred > 0.5).astype(int)

# Display prediction results
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.flatten()
})

# Add decision column
result["Decision"] = result["Predicted"].apply(
    lambda x: "Employee Will Leave" if x == 1 else "Employee Will Stay"
)

print(result.head(20))

# ANN model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
model.save("churn_model.h5")
print("Model saved successfully")
import pickle
with open ("scaler.pkl","wb") as f:
    pickle.dump(sc,f)