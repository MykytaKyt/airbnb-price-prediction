import click
import pandas as pd
from catboost import CatBoostRegressor
from click import secho


# Function to make predictions
def make_predictions(input_file, model_file, output_file):
    try:
        # Load the data for prediction
        df = pd.read_csv(input_file)

        # Load the trained model
        model = CatBoostRegressor()
        model.load_model(model_file)

        # Make predictions
        predictions = model.predict(df)

        # Create a DataFrame with predictions
        prediction_df = pd.DataFrame({'Predicted Price': predictions})

        # Save the predictions to an output file
        prediction_df.to_csv(output_file, index=False)

        secho("Predictions completed and saved to the output file.", fg="green")
    except Exception as e:
        secho(f"Error: {str(e)}", fg="red")


@click.command()
@click.option("--input-file", required=True, help="Input CSV file with data for prediction")
@click.option("--model-file", required=True, help="Trained model file path")
@click.option("--output-file", required=True, help="Output CSV file for predictions")
def main(input_file, model_file, output_file):
    make_predictions(input_file, model_file, output_file)


if __name__ == "__main__":
    main()
