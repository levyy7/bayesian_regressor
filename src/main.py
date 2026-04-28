import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.bayesian_linear_regressor import BayesianLinearRegressor

if __name__ == '__main__':
    # fetch dataset: Communities and Crime
    dataset = fetch_ucirepo(id=183)

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    # metadata
    print(dataset.metadata)

    # variable information
    print(dataset.variables)

    # Drop variables with lost values
    X = X.dropna(axis=1)
    # Drop categorical variables
    X = X.select_dtypes(include=[np.number])

    X = X.to_numpy()
    y = y.to_numpy()

    # Normalize Variabl;es
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = ((y - y.mean()) / y.std()).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size : {X_train.shape}")
    print(f"Test size  : {X_test.shape}")
    print(f"Features   : {X_train.shape[1]}")

    # Fit train data
    blr = BayesianLinearRegressor(sigma2=1.0, sigma2_v=1.0)
    blr.fit(X_train, y_train)

    # Predict test data
    mean_pred, var_pred = blr.predict(X_test)
    std_pred = np.sqrt(var_pred)

    print("\n--- Sample predictions (first 10 test points) ---")
    print(f"{'True y':>10}  {'Predicted μ':>12}  {'Std σ':>8}  {'|Error|':>8}")
    print("-" * 46)
    for i in range(10):
        print(f"{y_test[i]:>10.4f}  {mean_pred[i]:>12.4f}  {std_pred[i]:>8.4f}  {abs(y_test[i] - mean_pred[i]):>8.4f}")
