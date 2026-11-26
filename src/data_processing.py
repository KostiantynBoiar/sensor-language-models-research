from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm

class DataProcessing:
    ACTIVITY_CLASSES = [
        "sleep", "sitting", "standing", "household-chores",
        "manual-work", "walking", "mixed-activity",
        "vehicle", "sports", "bicycling",
    ]

    def __init__(self, data_dir, split_dir):
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir)
        self.ann_dict = self._load_annotation_dict()

    def _load_annotation_dict(self):
        path = self.data_dir / "annotation-label-dictionary.csv"
        df = pd.read_csv(path)
        return df.set_index("annotation")

    def load(self, participant_id: str):
        path = self.data_dir / f"{participant_id}.csv.gz"
        df = pd.read_csv(
            path,
            compression="gzip",
            dtype={"annotation": "str"},   # fixes the warning
            low_memory=False,
        )
        return df

    def resample_window(self, xyz_100hz: np.ndarray) -> np.ndarray:
            """
            Resample one window from 100 Hz → 30 Hz.
            Input:  (1000, 3)
            Output: (300, 3)
            """
            # Simple anti-alias + resample
            target_len = 300
            xyz_30hz = signal.resample(xyz_100hz, target_len, axis=0)
            return xyz_30hz.astype(np.float32)

    def map_label(self, raw_annotation: str):
        """Turn a raw annotation string into one of the 10 classes (or None)."""
        try:
            # You may need to adjust the column name after inspecting the file once
            mapped = self.ann_dict.loc[raw_annotation].iloc[0]
            if mapped in self.ACTIVITY_CLASSES:
                return mapped
        except KeyError:
            pass
        return None

    def extract_windows(self, df, window_sec=10, fs=100):
        """
        Very simple window extractor.
        Returns list of (window_array, label) pairs.
        """
        window_len = window_sec * fs
        n_windows = (len(df) - window_len) // window_len + 1
        windows = []

        for start in tqdm(range(0, len(df) - window_len + 1, window_len), total=n_windows, desc="Extracting windows"):
            chunk = df.iloc[start:start + window_len]
            label = self.map_label(chunk["annotation"].iloc[0])  # majority later if needed
            if label is None:
                continue
            xyz_100hz = chunk[["x", "y", "z"]].values.astype(np.float32)
            xyz_30hz = self.resample_window(xyz_100hz=xyz_100hz)
            
            windows.append((xyz_30hz, label))

            
        return windows

dp = DataProcessing(data_dir="data/capture24", split_dir="data/splits")
df = dp.load("P001")
windows = dp.extract_windows(df)
print(f"Got {len(windows)} windows")
print(windows[0][0].shape, windows[0][1])