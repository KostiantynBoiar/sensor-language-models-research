# scripts/fit_percentiles.py
from pathlib import Path
import numpy as np
from data_processing import DataProcessing

def main():
    dp = DataProcessing(
        data_dir="data/capture24",
        split_dir="data/splits",
    )

    # Use your actual training participant list
    # (or load it from the split file)
    train_pids = [f"P{i:03d}" for i in range(1, 81)]   # example: first 80

    all_features = []
    for pid in train_pids:
        print(f"Processing {pid}...")
        df = dp.load(pid)
        windows = dp.extract_windows(df)
        for window, _ in windows:
            feats = dp.compute_caption_features(window)
            all_features.append(feats)

    train_features = np.stack(all_features)
    print(f"Collected {train_features.shape[0]} windows")

    dp.fit_percentiles(train_features)

    # Save so you don't have to recompute every time
    np.save("data/splits/feature_percentiles.npy", dp.percentiles)
    print("Saved percentiles to data/splits/feature_percentiles.npy")

if __name__ == "__main__":
    main()