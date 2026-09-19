"""
One-time script: extract a static seller_id -> seller_state lookup table
from the database, so the inference service never needs a live database
connection to answer a single prediction request.

Run this once (and again only if the sellers table changes):
    python -m scripts.build_seller_lookup

This is training/offline tooling, not part of the inference pipeline --
that is why it lives in scripts/, not in src/.
"""

import pandas as pd
from sqlalchemy import create_engine

from src.config import CONFIG, PROJECT_ROOT


def main():
    db_cfg = CONFIG["database"]
    conn_string = (
        f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}"
        f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}"
    )
    engine = create_engine(conn_string)

    sellers = pd.read_sql_table("sellers", engine)
    seller_lookup = sellers[["seller_id", "seller_state"]].copy()

    output_path = PROJECT_ROOT / "data" / "sellers_lookup.parquet"
    seller_lookup.to_parquet(output_path, index=False)

    print(f"Saved {output_path} -> {seller_lookup.shape}")
    print(seller_lookup.head())


if __name__ == "__main__":
    main()
