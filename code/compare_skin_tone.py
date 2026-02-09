import os
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["bs", "fs", "wd", "pf"]

def find_csv(base_output_dir, mode):
    testing_dir = os.path.join(base_output_dir, "results", mode, "testing")
    filename = f"testing_avg_per_skin_tone_{mode}.csv"
    path = os.path.join(testing_dir, filename)
    return path if os.path.exists(path) else None

def load_mode_df(base_output_dir, mode):
    path = find_csv(base_output_dir, mode)
    if path is None:
        return None
    df = pd.read_csv(path)
    df["mode"] = mode
    return df

def combine_all(base_output_dir):
    dfs = []
    for m in MODES:
        df = load_mode_df(base_output_dir, m)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def save_combined(df, base_output_dir):
    out_dir = os.path.join(base_output_dir, "results", "comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "testing_avg_per_skin_tone_all_modes.csv")
    df.to_csv(out_csv, index=False)
    return out_csv

def plot_comparison(df, base_output_dir, metric="f1_macro_avg"):
    out_dir = os.path.join(base_output_dir, "results", "comparison")
    os.makedirs(out_dir, exist_ok=True)
    pv = df.pivot_table(index="skin_tone", columns="mode", values=metric)
    pv = pv.reindex(sorted(pv.index))
    ax = pv.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"Per skin tone comparison ({metric})")
    ax.set_ylabel(metric)
    ax.set_xlabel("skin_tone")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"comparison_{metric}.png")
    plt.savefig(out_png)
    return out_png

def main(base_output_dir=None):
    if base_output_dir is None:
        base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    df = combine_all(base_output_dir)
    if df is None:
        print("No testing_avg_per_skin_tone CSVs found for any mode.")
        return
    out_csv = save_combined(df, base_output_dir)
    print(f"Combined CSV saved: {out_csv}")
    img1 = plot_comparison(df, base_output_dir, metric="f1_macro_avg")
    print(f"Figure saved: {img1}")
    img2 = plot_comparison(df, base_output_dir, metric="accuracy_avg")
    print(f"Figure saved: {img2}")

if __name__ == "__main__":
    main()