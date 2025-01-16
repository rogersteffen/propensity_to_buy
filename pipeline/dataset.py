import duckdb
import polars as pl
import pyarrow
import os
from kaggle.api.kaggle_api_extended import KaggleApi

from IPython.display import display, HTML
from itables import init_notebook_mode

class KaggleDataset():

    def load(self):
        # There are other ways to download the files.
        # For this project I do not use the images (which take a long time)
        # Get a cup of coffee or change to one file at a time
        # Better than kaggle docs: https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python

        # not positive this is needed ...
        # os.environ['KAGGLE_CONFIG_DIR'] = '/Users/rogerteffen/.kaggle/'
        # print(os.environ['KAGGLE_CONFIG_DIR'])

        api = KaggleApi()
        api.authenticate()

        data_set = 'h-and-m-personalized-fashion-recommendations'

        # Download all files for a competition
        # Signature: competition_download_files(competition, path=None, force=False, quiet=True)
        # api.competition_download_files('h-and-m-personalized-fashion-recommendations',
        #                                path='../data')

        api.competition_download_file('h-and-m-personalized-fashion-recommendations',
                                      'articles.csv', path='../data')
        api.competition_download_file('h-and-m-personalized-fashion-recommendations',
                                      'customers.csv', path='../data')
        api.competition_download_file('h-and-m-personalized-fashion-recommendations',
                                      'transactions_train.csv', path='../data')

        # Define the directory containing the .zip files
        zip_dir = '../data'
        destination_dir = '../data'

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Iterate through all files in the directory
        for file in os.listdir(zip_dir):
            if file.endswith('.zip'):
                # Construct full path to the zip file
                zip_path = os.path.join(zip_dir, file)

                # Unzip the file
                os.system(f"unzip -o {zip_path} -d {destination_dir}")
                print(f"Unzipped: {file}")


class CSVDataset():

    def __init__(self, csv_path: str, csv_files: list):
        self.path = csv_path
        self.files = csv_files
        self.duckdb_conn = duckdb.connect()

    def load(self):
        for file in self.files:
            print(f"Loading {self.path + file}")
            table = file.replace(".csv", "")
            # todo: filename is a bit quirky - hardcoded cleanup
            table = table.replace("_train", "")
            self.load_file_into_view(self.path + file, table)
        return self.duckdb_conn

    def load_file_into_view(self, filename: str, viewname: str):
        # Load a CSV into a Polars DataFrame
        file_path = filename

        # todo: more hardcoding
        polars_df = pl.read_csv(file_path, schema_overrides={"t_dat": pl.Date, "article_id": pl.Utf8})

        # Register the Polars DataFrame as a DuckDB table
        # DuckDB can directly use Polars objects without conversion

        self.duckdb_conn.register(viewname, polars_df)

    def run_query_display_results(self, query: str):

        init_notebook_mode(all_interactive=True)

        # Execute the query and fetch results as an Arrow Table
        arrow_table = self.duckdb_conn.execute(query).fetch_arrow_table()

        # Convert the Arrow Table to a Polars DataFrame
        polars_df = pl.from_arrow(arrow_table)

        display(polars_df)

        return polars_df




