import toml
import pandas as pd
from sqlalchemy import create_engine
from tabulate import tabulate


# Load secrets from the TOML file
def load_secrets(file_path='../.streamlit/secrets.toml'):
    secrets = toml.load(file_path)
    return secrets['DSN']

# Query the database
def query_database(query, dsn, params=None):
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(dsn)
        # Execute the query
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error querying the database: {e}")
        return None

if __name__ == "__main__":
    # Load DSN from secrets
    dsn = load_secrets()

    # Query
    query = "SELECT * FROM predictions;"
    data = query_database(query, dsn)

    if data is not None:
        # Print results in a tabulated format
        print("\nQuery Results:\n")
        print(tabulate(data, headers="keys", tablefmt="grid", showindex=False))
