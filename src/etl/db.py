import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

#loads .env from the current directory (ideally project root)
load_dotenv()

def get_engine():
    """
    Create and return a SQLAlchemy engine for PostgreSQL using environment variables.
    """
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not dbname or not user or not password:
        raise ValueError(
            "Missing DB Credentials. Set DB_NAME, DB_USER, DB_PASSWORD in your .env file."
        )
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)

def get_schema() -> str:
    """
    Target schema in Postgres (defaults to public).
    """
    return os.getenv("DB_SCHEMA","public")