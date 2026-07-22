from sklearn.linear_model import LinearRegression
import numpy as np

# Extremely simple training data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
prediction = model.predict([[5]])
print("Prediction for input 5:", prediction[0])
