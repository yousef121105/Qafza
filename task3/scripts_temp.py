"""
One-time script: extract a static seller_id -> seller_state lookup table
from the database, so the inference service never needs a live DB
connection to compute the primary_seller_state feature.
"""

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://olist_user:olist_pass@localhost:5432/olist_db"
)

sellers = pd.read_sql_table("sellers", engine)
seller_lookup = sellers[["seller_id", "seller_state"]].copy()

seller_lookup.to_parquet("data/sellers_lookup.parquet", index=False)
print("Saved data/sellers_lookup.parquet ->", seller_lookup.shape)
print(seller_lookup.head())