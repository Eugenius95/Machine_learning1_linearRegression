from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = [
    [50, 70, 80], 
    [80, 85, 90], 
    [85, 87, 88], 
    [90, 95, 100],
    [60, 75, 85],   # New data point
    [70, 80, 90],   # New data point
    [75, 82, 85],   # New data point
    [95, 100, 105], # New data point
    [65, 78, 82],   # New data point
    [88, 92, 95],    # New data point
    [41, 30,10],
    [12000,11,10000]
]  
# Expanded target values
y = [65, 12, 98, 92, 70, 85, 90, 500, 75, 88, 14, 11]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Output the coefficients and intercept
print("Coefficients:", model.coef_)       
print("Intercept:", model.intercept_)

# Make predictions using the model
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual values:", y_test)
