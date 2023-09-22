import click
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import model_selection

import joblib
import os

@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
def main(input_file):
    try:
        click.secho("Loading the input DataFrame...", fg="cyan")
        df = pd.read_csv(input_file)
        click.secho("Input DataFrame loaded successfully.", fg="green")

        click.secho("Splitting data into train and test sets...", fg="cyan")
        X, y = df.iloc[:, df.columns != 'log_price'].values, df['log_price'].values
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

        click.secho("Training and evaluating models...", fg="cyan")
        model_folder = "model"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        train_and_evaluate_models(X_train, y_train, X_test, y_test, model_folder)

        click.secho("Training completed.", fg="green")

    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")

def train_and_evaluate_models(X_train, y_train, X_test, y_test, model_folder):
    cat = CatBoostRegressor(iterations=100, learning_rate=0.001, depth=10, l2_leaf_reg=100, verbose=0)
    xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    lgbm = LGBMRegressor(random_state=37, force_col_wise=True)
    regr = RandomForestRegressor(max_depth=2, random_state=0)

    cat.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    regr.fit(X_train, np.log(y_train))

    preds_cat = cat.predict(X_test)
    preds_xgb = xgb.predict(X_test)
    preds_lgbm = lgbm.predict(X_test)
    preds_regr = np.exp(regr.predict(X_test))

    # Evaluate models
    mse_cat = mean_squared_error(preds_cat, y_test)
    mae_cat = mean_absolute_error(preds_cat, y_test)
    mse_xgb = mean_squared_error(preds_xgb, y_test)
    mae_xgb = mean_absolute_error(preds_xgb, y_test)
    mse_lgbm = mean_squared_error(preds_lgbm, y_test)
    mae_lgbm = mean_absolute_error(preds_lgbm, y_test)
    mse_regr = mean_squared_error(preds_regr, y_test)
    mae_regr = mean_absolute_error(preds_regr, y_test)

    ensemble_model = StackingRegressor([
        ("catboost", cat),
        ("random forest", regr),
        ('xgboost', xgb),
        ('lightgbm', lgbm)
    ], final_estimator=XGBRegressor())

    ensemble_model.fit(X_train, y_train)
    preds_ensemble = ensemble_model.predict(X_test)
    mse_ensemble = mean_squared_error(preds_ensemble, y_test)
    mae_ensemble = mean_absolute_error(preds_ensemble, y_test)

    best_model = None
    if mse_cat <= mse_xgb and mse_cat <= mse_lgbm and mse_cat <= mse_regr and mse_cat <= mse_ensemble:
        best_model = cat
    elif mse_xgb <= mse_cat and mse_xgb <= mse_lgbm and mse_xgb <= mse_regr and mse_xgb <= mse_ensemble:
        best_model = xgb
    elif mse_lgbm <= mse_cat and mse_lgbm <= mse_xgb and mse_lgbm <= mse_regr and mse_lgbm <= mse_ensemble:
        best_model = lgbm
    elif mse_regr <= mse_cat and mse_regr <= mse_xgb and mse_regr <= mse_lgbm and mse_regr <= mse_ensemble:
        best_model = regr
    else:
        best_model = ensemble_model

    click.secho(f"Best model selected: {type(best_model).__name__}", fg="green")
    click.secho(f"Best model MSE: {mse_cat:.4f}", fg="green")
    click.secho(f"Best model MAE: {mae_cat:.4f}", fg="green")

    model_filename = f"{model_folder}/best_model.joblib"
    joblib.dump(best_model, model_filename)
    click.secho(f"Best model saved to {model_filename}", fg="green")

    result_table = pd.DataFrame({
        'Model': ['CatBoost', 'XGBoost', 'LightGBM', 'Random Forest', 'Ensemble'],
        'MSE': [mse_cat, mse_xgb, mse_lgbm, mse_regr, mse_ensemble],
        'MAE': [mae_cat, mae_xgb, mae_lgbm, mae_regr, mae_ensemble]
    })
    click.secho("Model Comparison:", fg="cyan")
    click.echo(result_table.to_string(index=False))

if __name__ == "__main__":
    main()
