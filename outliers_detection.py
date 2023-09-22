import click
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
@click.option("--output-file", required=True, help="Output CSV file path for cleaned DataFrame")
def main(input_file, output_file):
    try:
        click.secho("Loading the input DataFrame...", fg="cyan")
        # Load the input DataFrame
        df = pd.read_csv(input_file)
        click.secho("Input DataFrame loaded successfully.", fg="green")

        click.secho("Performing Isolation Forest Outlier Detection...", fg="cyan")
        isolation_forest = IsolationForest(n_estimators=200, max_samples='auto', contamination=0.01)  # Adjust contamination value
        isolation_forest.fit(df[['log_price']].values.reshape(-1, 1))
        isolation_forest_outlier = isolation_forest.predict(df[['log_price']].values.reshape(-1, 1))

        click.secho("Performing One-Class SVM Outlier Detection...", fg="cyan")
        model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.01)  # Adjust gamma and nu values
        oneclass_svm_outlier = model.fit_predict(df[['log_price']].values.reshape(-1, 1))

        click.secho("Performing Z-Score Outlier Detection...", fg="cyan")
        zscore = (df['log_price'] - df['log_price'].mean()) / df['log_price'].std()
        zscore_outlier = np.abs(zscore) > 1.0  # Adjust the Z-Score threshold

        click.secho("Performing Tukey's Method Outlier Detection...", fg="cyan")
        Q1 = df['log_price'].quantile(0.25)
        Q3 = df['log_price'].quantile(0.75)
        IQR = Q3 - Q1
        tukey_outlier = (df['log_price'] < (Q1 - 1.5 * IQR)) | (
                    df['log_price'] > (Q3 + 1.5 * IQR))  # Adjust the multiplier

        combined_outlier = (
            isolation_forest_outlier +
            oneclass_svm_outlier +
            zscore_outlier +
            tukey_outlier
        ) >= 3

        click.secho(f"Number of values identified as outliers: {np.sum(combined_outlier)}", fg="yellow")

        click.secho("Removing Outliers...", fg="cyan")
        df = df[~combined_outlier]
        df = df.loc[df['log_price'] > 1]
        df = df.astype(int)
        click.secho(f"Number of values remaining: {df.shape[0]}", fg="green")
        click.secho("Outliers removed from the DataFrame.", fg="green")

        click.secho(f"Saving DataFrame without Outliers to {output_file}...", fg="cyan")
        df.to_csv(output_file, index=False)
        click.secho("DataFrame without Outliers saved successfully.", fg="green")

    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")

if __name__ == "__main__":
    main()
