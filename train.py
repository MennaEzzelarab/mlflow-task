import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

seed = 42

data_df = fetch_california_housing(as_frame=True)
X = data_df.data
y = data_df.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

configs = [
    {"name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

mlflow.set_experiment("House Price Prediction")

for conf in configs:
    with mlflow.start_run(run_name=conf["name"]):
        model = GradientBoostingRegressor(
            max_depth = conf["max_depth"],
            learning_rate = conf["learning_rate"],
            random_state = seed
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mlflow.log_param("max_depth", conf["max_depth"])
        mlflow.log_param("learning_rate", conf["learning_rate"])
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"{conf['name']}: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
