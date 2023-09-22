import click
import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm
from click import secho

def process_text(input_file, output_file):
    try:
        secho("Loading the dataset...", fg="cyan")
        # Load the dataset
        df = pd.read_csv(input_file)
        secho("Dataset loaded successfully.", fg="green")

        secho("Initializing KeyBERT model...", fg="cyan")
        kw_model = KeyBERT()
        secho("KeyBERT model initialized.", fg="green")

        secho("Extracting keywords for each description...", fg="cyan")
        keywords_list = []
        for i in tqdm(range(len(df.description))):
            keys = kw_model.extract_keywords(df.description[i], keyphrase_ngram_range=(1, 1), stop_words='english',
                                             top_n=3)
            keys = [i[0] for i in keys]
            keywords_list.append(keys)
        secho("Keywords extracted successfully.", fg="green")

        key_list = [item for sublist in keywords_list for item in sublist]

        secho("Creating binary columns for unique keywords...", fg="cyan")
        unique_keys = set(key_list)
        key_categorical = pd.DataFrame(
            data=[[1 if key in desc else 0 for key in unique_keys] for desc in df.description], columns=list(unique_keys))
        secho("Binary columns created for keywords.", fg="green")

        secho("Calculating Spearman correlation with log_price...", fg="cyan")
        k = key_categorical.corrwith(df["log_price"], method='spearman').sort_values(ascending=False)
        column_names = k[k > 0.09].index.tolist()
        secho("Spearman correlation calculated.", fg="green")

        if 'bedrooms' in column_names:
            column_names.remove('bedrooms')

        secho("Selecting relevant columns...", fg="cyan")
        key_categorical = key_categorical[column_names]
        secho("Relevant columns selected.", fg="green")

        secho("Concatenating new columns to the DataFrame...", fg="cyan")
        df = pd.concat([df, key_categorical], axis=1)
        secho("New columns concatenated.", fg="green")

        secho("Dropping single and text columns...", fg="cyan")
        cat_list = ['description']
        df = df.drop(cat_list, axis=1)
        secho("Columns dropped.", fg="green")

        secho("Saving the processed data...", fg="cyan")
        df.to_csv(output_file, index=False)
        secho("Processed data saved successfully.", fg="green")

        secho("Text processing completed.", fg="green")
    except Exception as e:
        secho(f"Error: {str(e)}", fg="red")

@click.command()
@click.option("--input-file", required=True, help="Input CSV file path")
@click.option("--output-file", required=True, help="Output CSV file path after text processing")
def main(input_file, output_file):
    process_text(input_file, output_file)

if __name__ == "__main__":
    main()
