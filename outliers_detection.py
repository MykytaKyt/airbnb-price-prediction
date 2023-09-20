import click
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from click import secho


# Function to detect outliers
def detect_outliers(input_file, output_file):
    try:
        # Load the preprocessed data
        df = pd.read_csv(input_file)

        # Initialize Isolation Forest model
        isolation_forest = IsolationForest(n_estimators=100)

        # Fit Isolation Forest to detect outliers
        outlier_labels = isolation_forest.fit_predict(df[['price']])

        # Identify outlier indices
        outlier_indices = df.index[outlier_labels == -1]

        # Remove outliers from the dataframe
        df_no_outliers = df.drop(outlier_indices)

        # Initialize DBSCAN model
        db = DBSCAN(eps=250, min_samples=9, metric='euclidean')

        # Fit DBSCAN to detect outliers
        db_labels = db.fit_predict(df_no_outliers.values)

        # Identify outlier indices
        db_outlier_indices = df_no_outliers.index[db_labels == -1]

        # Remove DBSCAN outliers from the dataframe
        df_no_db_outliers = df_no_outliers.drop(db_outlier_indices)

        # Perform t-SNE visualization of the data
        X_embedded = TSNE(n_components=2).fit_transform(df_no_db_outliers.values)

        # Save the data without outliers
        df_no_db_outliers.to_csv(output_file, index=False)

        secho("Outlier detection completed.", fg="green")
    except Exception as e:
        secho(f"Error: {str(e)}", fg="red")


@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
@click.option("--output-file", required=True, help="Output CSV file path after outlier detection")
def main(input_file, output_file):
    detect_outliers(input_file, output_file)


if __name__ == "__main__":
    main()
