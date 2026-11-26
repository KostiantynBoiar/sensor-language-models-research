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
    LEVELS = ["very low", "low", "medium", "high", "very high"]

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

    def _statistical_features(self, channels) -> list:
        """24 features: mean, std, min, max of each of the 6 channels."""
        features = []
        for ch in channels:
            features.extend([
                float(ch.mean()),
                float(ch.std()),
                float(ch.min()),
                float(ch.max()),
            ])
        return features

    def _structural_features(self, channels, magnitude) -> list:
        """10 structural features."""
        features = []
        t = np.arange(len(magnitude))

        # Trend (slope) of each channel
        for ch in channels:
            slope = np.polyfit(t, ch, 1)[0]
            features.append(float(slope))

        # Peaks and drops in overall movement
        mag = magnitude
        peaks = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:]))[0]
        drops = np.where((mag[1:-1] < mag[:-2]) & (mag[1:-1] < mag[2:]))[0]
        features.append(float(len(peaks)))
        features.append(float(len(drops)))

        # Dominant frequency + power share
        fft = np.fft.rfft(mag)
        power = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(mag), d=1/30)
        if power.sum() > 0:
            dom_idx = np.argmax(power[1:]) + 1
            features.append(float(freqs[dom_idx]))
            features.append(float(power[dom_idx] / power.sum()))
        else:
            features.extend([0.0, 0.0])

        return features

    def _event_features(self, magnitude, pitch) -> list:
        """3 binary event features."""
        # Still period
        still = float(np.mean(np.abs(magnitude) < 0.05) > 0.7)

        # Large change in wrist angle
        angle_change = float(np.max(np.abs(np.diff(pitch))) > np.deg2rad(30))

        # Sustained rhythmic movement
        fft = np.fft.rfft(magnitude)
        power = np.abs(fft)**2
        freqs = np.fft.rfftfreq(len(magnitude), d=1/30)
        dom_idx = np.argmax(power[1:]) + 1 if power.sum() > 0 else 0
        rhythmic = float(
            (0.5 < freqs[dom_idx] < 3.0) and (power[dom_idx] / (power.sum() + 1e-8) > 0.3)
        )

        return [still, angle_change, rhythmic]

    def fit_percentiles(self, train_features: np.ndarray):
        """
        train_features: (N, 37) computed on training participants
        Stores the 4 cut points for each of the 34 numeric features.
        """
        numeric = train_features[:, :34]
        # 20, 40, 60, 80 percentiles
        self.percentiles = np.percentile(numeric, [20, 40, 60, 80], axis=0)
        print("Percentiles fitted.")

    def _value_to_level(self, value: float, feature_idx: int) -> str:
        cuts = self.percentiles[:, feature_idx]   # shape (4,)
        level_idx = np.searchsorted(cuts, value)
        return self.LEVELS[level_idx]

    def features_to_statements(self, features: np.ndarray) -> list[str]:
        """
        Convert the 37 caption features into a list of statements.
        features: (37,) array
        returns: list of statements (roughly 35)
        """

        # Templates for the 34 numeric features
        # Order matches the order we computed them:
        # 24 statistical (mean/std/min/max × 6 channels) + 10 structural
        templates = [
            # statistical: x, y, z, magnitude, pitch, roll 
            "The mean of the x-axis is {level}.",
            "The variability of the x-axis is {level}.",
            "The minimum of the x-axis is {level}.",
            "The maximum of the x-axis is {level}.",

            "The mean of the y-axis is {level}.",
            "The variability of the y-axis is {level}.",
            "The minimum of the y-axis is {level}.",
            "The maximum of the y-axis is {level}.",

            "The mean of the z-axis is {level}.",
            "The variability of the z-axis is {level}.",
            "The minimum of the z-axis is {level}.",
            "The maximum of the z-axis is {level}.",

            "The mean of overall movement is {level}.",
            "The variability of overall movement is {level}.",
            "The minimum of overall movement is {level}.",
            "The maximum of overall movement is {level}.",

            "The mean of wrist pitch is {level}.",
            "The variability of wrist pitch is {level}.",
            "The minimum of wrist pitch is {level}.",
            "The maximum of wrist pitch is {level}.",

            "The mean of wrist roll is {level}.",
            "The variability of wrist roll is {level}.",
            "The minimum of wrist roll is {level}.",
            "The maximum of wrist roll is {level}.",

            # structural
            "The trend of the x-axis is {level}.",
            "The trend of the y-axis is {level}.",
            "The trend of the z-axis is {level}.",
            "The trend of overall movement is {level}.",
            "The trend of wrist pitch is {level}.",
            "The trend of wrist roll is {level}.",

            "The number of movement peaks is {level}.",
            "The number of movement drops is {level}.",

            "The dominant frequency of overall movement is {level}.",
            "The power share of the dominant frequency is {level}.",
        ]

        statements = []

        # 34 numeric features → 5 levels (crude binning for now)
        numeric = features[:34]
        for i, val in enumerate(numeric):
            # Simple mapping into 0–4; replace with real percentiles later
            level_idx = int(np.clip(np.round(val * 2 + 2), 0, 4))
            statements.append(templates[i].format(level=self.LEVELS[level_idx]))

        # 3 events (binary)
        if features[34] > 0.5:
            statements.append("There is a still period.")
        if features[35] > 0.5:
            statements.append("There is a large change in wrist angle.")
        if features[36] > 0.5:
            statements.append("There is sustained rhythmic movement.")

        return statements
    
    def compute_caption_features(self, window: np.ndarray) -> np.ndarray:
        """
        window: (300, 3) at 30 Hz
        returns: (37,) feature vector
        """
        x, y, z = window[:, 0], window[:, 1], window[:, 2]

        magnitude = np.sqrt(x**2 + y**2 + z**2) - 1.0
        pitch = np.arctan2(y, np.sqrt(x**2 + z**2))
        roll  = np.arctan2(-x, z)
        channels = [x, y, z, magnitude, pitch, roll]

        features = []
        features += self._statistical_features(channels)
        features += self._structural_features(channels, magnitude)
        features += self._event_features(magnitude, pitch)

        return np.array(features, dtype=np.float32)

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
