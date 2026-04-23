# ========================================
# DATA INGESTION MODULE
# ========================================
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data"
INGESTED_DIR = BASE_DIR / "pipeline" / "ingested"

INPUT_FILE_FEATS = RAW_DIR / "A.csv"
INPUT_FILE_Y = RAW_DIR / "A_targets.csv"
OUTPUT_FILE = INGESTED_DIR / "data.csv"


def ingest_data():
    """
    Load features and targets from separate CSV files (Dataset A),
    merge on Student_ID, validate, and save combined dataset.
    """
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("STEP 1: DATA INGESTION")
    print("=" * 50)

    feat = pd.read_csv(INPUT_FILE_FEATS)
    target = pd.read_csv(INPUT_FILE_Y)

    print(f"Features loaded : {feat.shape}")
    print(f"Targets loaded  : {target.shape}")

    df = pd.merge(feat, target, on='Student_ID')

    assert not df.empty, "Merged dataset is empty"
    assert df.shape[0] == feat.shape[0], "Row count mismatch after merge"

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged dataset  : {df.shape}")
    print(f"Saved to        : {OUTPUT_FILE}")
    print("")

    return df


if __name__ == "__main__":
    ingest_data()