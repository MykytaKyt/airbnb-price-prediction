import click
import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm
from click import secho


# Function to perform text processing
def process_text(input_file, output_file):
    try:
        # Load the dataset
        df = pd.read_csv(input_file)

        # Initialize KeyBERT model
        kw_model = KeyBERT()

        # Extract keywords for each description
        keywords_list = []
        for i in tqdm(range(len(df.description))):
            keys = kw_model.extract_keywords(df.description[i], keyphrase_ngram_range=(1, 1), stop_words='english',
                                             top_n=3)
            keys = [i[0] for i in keys]
            keywords_list.append(keys)

        # Flatten the list of keywords
        key_list = [item for sublist in keywords_list for item in sublist]

        # Create binary columns for unique keywords
        unique_keys = set(key_list)
        key_categorical = pd.DataFrame(
            data=[[1 if key in desc else 0 for key in unique_keys] for desc in df.description], columns=unique_keys)

        # Concatenate the new columns to the DataFrame
        df = pd.concat([df, key_categorical], axis=1)

        # Drop single and text columns
        cat_list = ['name', 'summary', 'description', 'host_location', 'host_about', 'host_neighbourhood',
                    'host_verifications', 'neighbourhood_cleansed', 'market', 'country_code', 'country', 'host_since',
                    'neighborhood_overview', 'transit', 'host_since']
        df = df.drop(cat_list, axis=1)

        # Save the processed data
        df.to_csv(output_file, index=False)

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
