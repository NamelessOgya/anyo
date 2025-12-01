import pandas as pd
from pathlib import Path

data_dir = Path("data/ml-100k")
movies_path = data_dir / "movies.csv"
movies_df = pd.read_csv(movies_path)
print(f"Total items: {len(movies_df)}")

# Check for "unknown" in titles
unknowns = movies_df[movies_df['title'].str.contains("unknown", case=False, na=False)]
print(f"Items with 'unknown' in title: {len(unknowns)}")
if len(unknowns) > 0:
    print(unknowns)

# Check for NaN titles
nans = movies_df[movies_df['title'].isna()]
print(f"Items with NaN title: {len(nans)}")
if len(nans) > 0:
    print(nans)

# Check for empty titles
empties = movies_df[movies_df['title'] == ""]
print(f"Items with empty title: {len(empties)}")
if len(empties) > 0:
    print(empties)
