import click
import pandas as pd
from click import secho


def preprocess_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)

        df["log_price"] = df["log_price"].replace('[\$,]', '', regex=True).astype(float)
        df["cleaning_fee"] = df["cleaning_fee"].replace('[\$,]', '', regex=True).astype(float)

        df['host_response_rate'] = df['host_response_rate'].astype(str)
        df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').apply(pd.to_numeric, errors='coerce')

        df['city'] = df['city'].replace({'Washington, D.C.': 'Washington'})
        df['city'] = df['city'].replace({'Washington ': 'Washington'})

        df.to_csv(output_file, index=False)

        secho("Data preprocessing completed.", fg="green")
    except Exception as e:
        secho(f"Error: {str(e)}", fg="red")


@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
@click.option("--output-file", required=True, help="Output CSV file path after preprocessing")
def main(input_file, output_file):
    preprocess_data(input_file, output_file)


if __name__ == "__main__":
    main()
