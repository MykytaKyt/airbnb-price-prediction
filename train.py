import click
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from click import secho


# Function to train the model
def train_model(input_file, output_model):
    try:
        # Load the data without outliers
        df = pd.read_csv(input_file)

        # Split the data into features and target
        X = df.drop('price', axis=1)
        y = df['price']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Initialize and train the models
        cat = CatBoostRegressor(iterations=100, learning_rate=0.001, depth=10, l2_leaf_reg=100)
        cat.fit(X_train, y_train)

        xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        xgb.fit(X_train, y_train)

        lgbm = LGBMRegressor(random_state=37)
        lgbm.fit(X_train, y_train)

        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X_train, y_train)

        ensemble_model = StackingRegressor([
            ("catboost", cat),
            ("random forest", regr),
            ('xgboost', xgb),
            ('lightgbm', lgbm)
        ], final_estimator=XGBRegressor())
        ensemble_model.fit(X_train, y_train)

        # Evaluate models on the test set
        cat_preds = cat.predict(X_test)
        cat_rmse = mean_squared_error(cat_preds, y_test, squared=False)
        cat_mae = mean_absolute_error(cat_preds, y_test)

        xgb_preds = xgb.predict(X_test)
        xgb_rmse = mean_squared_error(xgb_preds, y_test, squared=False)
        xgb_mae = mean_absolute_error(xgb_preds, y_test)

        lgbm_preds = lgbm.predict(X_test)
        lgbm_rmse = mean_squared_error(lgbm_preds, y_test, squared=False)
        lgbm_mae = mean_absolute_error(lgbm_preds, y_test)

        ensemble_preds = ensemble_model.predict(X_test)
        ensemble_rmse = mean_squared_error(ensemble_preds, y_test, squared=False)
        ensemble_mae = mean_absolute_error(ensemble_preds, y_test)

        # Save the best model
        best_model = cat if cat_rmse < min(xgb_rmse, lgbm_rmse, ensemble_rmse) else (
            xgb if xgb_rmse < min(lgbm_rmse, ensemble_rmse) else (
                lgbm if lgbm_rmse < ensemble_rmse else ensemble_model))
        best_model.save_model(output_model)

        secho(f"Model training completed. Best model: {type(best_model).__name__}", fg="green")
        secho(f"CatBoost RMSE: {cat_rmse}, MAE: {cat_mae}", fg="green")
        secho(f"XGBoost RMSE: {xgb_rmse}, MAE: {xgb_mae}", fg="green")
        secho(f"LGBM RMSE: {lgbm_rmse}, MAE: {lgbm_mae}", fg="green")
        secho(f"Ensemble RMSE: {ensemble_rmse}, MAE: {ensemble_mae}", fg="green")
    except Exception as e:
        secho(f"Error: {str(e)}", fg="red")


@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
@click.option("--output-model", required=True, help="Output model file path")
def main(input_file, output_model):
    train_model(input_file, output_model)


if __name__ == "__main__":
    main()
