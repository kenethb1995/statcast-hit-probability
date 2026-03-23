"""
Load Statcast batted-ball events into Postgres, skipping days already loaded.
"""

from sqlalchemy import text
from .db import get_engine, get_schema
from .pull_statcast import pull_statcast_batted_balls
from datetime import datetime, timedelta

TABLE_NAME = "batted_ball_events"

def _qualified(schema: str) -> str:
    #Quote identifiers so schema/table are always safe in SQL
    return f'"{schema}"."{TABLE_NAME}"'

def row_count(engine, schema: str) -> int:
    with engine.connect() as conn:
        res = conn.execute(text(f'SELECT COUNT(*) FROM {_qualified(schema)};'))
        return int(res.scalar_one())

def date_range(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    curr = start
    while curr <= end:
        yield curr.strftime("%Y-%m-%d")
        curr += timedelta(days=1)

def rows_exist_for_date(engine, schema: str, game_date: str) -> bool:
    with engine.connect() as conn:
        res = conn.execute(
            text(f"""
            SELECT 1
            FROM {_qualified(schema)}
            WHERE game_date = :game_date
            LIMIT 1;
            """),
            {"game_date":game_date},
        )
        return res.first() is not None

def load_batted_ball_events(start_date: str, end_date: str) -> int:
    """
    Pull batted ball events for selected date range and append into Postgres

    Idempotent behavior:
        - if any rows already exist for the date range, the load is skipped for day
    """
    engine = get_engine()
    schema = get_schema()

    total_inserted = 0

    for day in date_range(start_date, end_date):
        if rows_exist_for_date(engine, schema, day):
            print(f"{day} already loaded - skipping.")
            continue

        df = pull_statcast_batted_balls(day, day)
        if df.empty:
            print(f"{day}: no rows returned.")
            continue

        before = row_count(engine, schema)

        df.to_sql(
            name=TABLE_NAME,
            con=engine,
            schema=schema,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=5000,
        )

        after = row_count(engine, schema)
        inserted = len(df)
        total_inserted += inserted
        change = after - before

        print(f"{day}: loaded {inserted} rows (change: {change})")

    print(f"Total inserted for {start_date} to {end_date}: {total_inserted}")
    return total_inserted

if __name__ == "__main__":
    #single-day load (day-by-day pipeline)
    load_batted_ball_events("2021-01-01", "2025-12-31")

