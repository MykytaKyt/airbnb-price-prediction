import click
import pandas as pd
from sklearn.impute import KNNImputer

def get_columns_with_missing_values(df):
    columns_with_missing_values = df.columns[df.isnull().any()].tolist()
    return columns_with_missing_values

@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
@click.option("--output-file", required=True, help="Output CSV file path for processed data")
def main(input_file, output_file):
    try:
        click.secho("Loading the input DataFrame...", fg="cyan")
        df = pd.read_csv(input_file)
        click.secho("Input DataFrame loaded successfully.", fg="green")

        click.secho("Processing 'amenities' column...", fg="cyan")
        df['amenities'] = df['amenities'].apply(
            lambda x: [] if x == '{}' else sorted(list(map(lambda x: x.replace('"', ''), x[1:-1].split(',')))))
        click.secho("'amenities' column processed successfully.", fg="green")

        click.secho("Extracting unique amenities...", fg="cyan")
        unique_amenities = sorted(set(amenity for amenities in df['amenities'] for amenity in amenities))
        click.secho("Unique amenities extracted successfully.", fg="green")

        click.secho("Creating binary columns for amenities...", fg="cyan")
        amenities_columns = unique_amenities
        amenities_rows = []
        for item in df['amenities']:
            tmp_list = []
            for amenity in unique_amenities:
                tmp_list.append(1 if amenity in item else 0)
            amenities_rows.append(tmp_list)

        amenities_categorical = pd.DataFrame(data=amenities_rows, columns=amenities_columns)
        click.secho("Binary columns for amenities created.", fg="green")

        click.secho("Concatenating amenities columns...", fg="cyan")
        df = pd.concat([df, amenities_categorical], axis=1)
        df = df.drop('amenities', axis=1)
        click.secho("Amenities columns concatenated.", fg="green")

        click.secho("Performing one-hot encoding for categorical columns...", fg="cyan")
        df = pd.get_dummies(df)
        click.secho("One-hot encoding completed.", fg="green")

        click.secho("Generating the list of columns with missing values...", fg="cyan")
        list_of_na = get_columns_with_missing_values(df)
        click.secho("List of columns with missing values generated.", fg="green")

        click.secho("Performing KNN imputation for missing values...", fg="cyan")
        impute = KNNImputer(n_neighbors=7)
        for col in list_of_na:
            df[[col]] = impute.fit_transform(df[[col]])
        click.secho("KNN imputation completed.", fg="green")

        click.secho("Dropping rows with any remaining missing values...", fg="cyan")
        df = df.dropna()
        click.secho("Rows with missing values dropped.", fg="green")

        click.secho("Saving the processed data to the specified output file...", fg="cyan")
        df.to_csv(output_file, index=False)
        click.secho(f"Processed data saved to {output_file}.", fg="green")

    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")

if __name__ == "__main__":
    main()
