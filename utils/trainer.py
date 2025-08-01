from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(model_name, X_train, X_test, y_train, y_test):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "KNN":
        model = KNeighborsRegressor()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor()
    else:
        return {"error": "Model tidak dikenali"}

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "R2 Score": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
    }
