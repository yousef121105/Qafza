import sqlite3

db = "mlflow-docker.db"

conn = sqlite3.connect(db)

old_path = "file:///C:/Users/TOKYO TECH/olist-mlops/task3/mlruns"
new_path = "file:///app/mlruns"

# Fix experiment artifact locations
conn.execute(
    """
    UPDATE experiments
    SET artifact_location = REPLACE(
        artifact_location,
        ?,
        ?
    )
    """,
    (old_path, new_path),
)

# Fix run artifact locations
conn.execute(
    """
    UPDATE runs
    SET artifact_uri = REPLACE(
        artifact_uri,
        ?,
        ?
    )
    """,
    (old_path, new_path),
)

# Fix registered model version sources
conn.execute(
    """
    UPDATE model_versions
    SET source = REPLACE(
        source,
        ?,
        ?
    )
    """,
    (old_path, new_path),
)

conn.commit()

print("Experiments:")
print(
    conn.execute("SELECT experiment_id, artifact_location FROM experiments").fetchall()
)

print("\nRuns:")
print(conn.execute("SELECT run_uuid, artifact_uri FROM runs").fetchall())

print("\nModel versions:")
print(conn.execute("SELECT name, source, run_id FROM model_versions").fetchall())

conn.close()
