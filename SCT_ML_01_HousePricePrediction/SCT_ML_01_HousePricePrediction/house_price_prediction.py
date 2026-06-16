import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset/Housing.csv")

# Features and target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
sample_house = pd.DataFrame({
    "area": [2000],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(sample_house)

print("\nPredicted Price:")
print(predicted_price[0])

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.savefig("screenshots/result_graph.png")
plt.show()