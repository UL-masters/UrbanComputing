import pandas as pd
import glob

def load_prsa(data_path):
    files = glob.glob(data_path + "/*.csv")
    
    dfs = []
    for f in files:
        station = f.split("_")[2]   # name between first underscores
        df = pd.read_csv(f)

        # Build a datetime column
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" +
            df["month"].astype(str) + "-" +
            df["day"].astype(str) + " " +
            df["hour"].astype(str) + ":00:00"
        )

        df["country"] = station   # map station → country label
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    
    return df
