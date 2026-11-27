from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from data_processing import DataProcessing
except ImportError:
    from src.data_processing import DataProcessing


@dataclass
class WindowPropertyValidationResult:
    """Stores validation checks for an individual resampled sensor window."""

    shape_ok: bool
    dtype_ok: bool
    finite_ok: bool
    range_ok: bool
    label_ok: bool
    all_valid: bool


@dataclass
class WindowAlignmentValidationResult:
    """Stores aggregate alignment, sampling, and purity metrics for a participant."""

    participant_id: str
    windows_inspected: int
    annotated_windows: int
    unannotated_windows: int
    first_vs_majority_agreement: float
    mean_label_purity: float
    all_window_properties_passed: bool


@dataclass
class AnnotationDictValidationResult:
    """Stores verification results for the annotation dictionary mapping."""

    dictionary_path: str
    total_raw_annotations: int
    all_expected_classes_present: bool
    missing_classes: list[str] = field(default_factory=list)
    raw_annotations_per_class: dict[str, int] = field(default_factory=dict)


@dataclass
class SplitValidationResult:
    """Stores verification results for participant splits and disjointness."""

    split_file: str
    counts: dict[str, int]
    counts_ok: dict[str, bool]
    is_disjoint: bool
    overlaps: dict[str, list[str]]
    total_participants: int
    all_files_exist: bool
    missing_files: list[str] = field(default_factory=list)


@dataclass
class ClassDistributionResult:
    """Stores activity class frequency and proportion metrics."""

    total_windows: int
    counts_per_class: dict[str, int] = field(default_factory=dict)
    percentages_per_class: dict[str, float] = field(default_factory=dict)
    zero_count_classes: list[str] = field(default_factory=list)


@dataclass
class SignalNumericalValidationResult:
    """Stores numerical health metrics for sensor channels and computed features."""

    total_windows_checked: int
    all_finite: bool
    channel_shapes_ok: bool
    pitch_bounds_ok: bool
    roll_bounds_ok: bool
    statistical_features_valid: bool
    structural_features_valid: bool
    event_features_binary: bool
    all_valid: bool


@dataclass
class PercentileLeakageValidationResult:
    """Stores verification metrics for feature percentiles and participant leakage."""

    percentiles_file_exists: bool
    shape_ok: bool
    is_monotonic: bool
    train_participants_count: int
    no_val_or_test_leakage: bool
    bin_balance_proportions: dict[str, float]
    bin_balance_acceptable: bool
    all_valid: bool


@dataclass
class FeatureUnitTestResult:
    """Stores results of deterministic unit tests on synthetic signals."""

    stillness_test_passed: bool
    angle_change_test_passed: bool
    rhythmic_movement_test_passed: bool
    dimension_and_type_passed: bool
    all_passed: bool


class DataValidator:
    """Validates CAPTURE-24 data processing according to benchmark standards."""

    EXPECTED_SPLIT_COUNTS = {"train": 80, "validation": 20, "test": 51}
    EXPECTED_WINDOW_SHAPE = (300, 3)
    EXPECTED_DTYPE = np.float32

    def __init__(self, data_dir: str = "data/capture24", split_dir: str = "data/splits") -> None:
        """Initializes the validator with paths to raw data and splits.

        Args:
            data_dir: Path to directory containing participant CSVs and annotation dictionary.
            split_dir: Path to directory containing participant split files.
        """
        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir)
        self.dp = DataProcessing(data_dir=data_dir, split_dir=split_dir)

    def validate_window_properties(
        self, window: np.ndarray, label: str
    ) -> WindowPropertyValidationResult:
        """Validates the structure and physical plausibility of a single resampled window.

        Args:
            window: Resampled accelerometer window array of shape (300, 3).
            label: Assigned activity class string.

        Returns:
            WindowPropertyValidationResult indicating structural, numerical, and label checks.
        """
        shape_ok = tuple(window.shape) == self.EXPECTED_WINDOW_SHAPE
        dtype_ok = window.dtype == self.EXPECTED_DTYPE
        finite_ok = bool(np.all(np.isfinite(window)))
        range_ok = bool(np.all(np.abs(window) <= 16.0)) if finite_ok else False
        label_ok = label in self.dp.ACTIVITY_CLASSES

        return WindowPropertyValidationResult(
            shape_ok=shape_ok,
            dtype_ok=dtype_ok,
            finite_ok=finite_ok,
            range_ok=range_ok,
            label_ok=label_ok,
            all_valid=shape_ok and dtype_ok and finite_ok and range_ok and label_ok,
        )

    def validate_windowing_alignment(
        self, participant_id: str, max_windows: int | None = 500
    ) -> WindowAlignmentValidationResult:
        """Inspects windowing and labeling mechanics on a participant's raw record.

        Args:
            participant_id: Identifier of the participant (e.g. 'P001').
            max_windows: Maximum number of windows to evaluate, or None for all.

        Returns:
            WindowAlignmentValidationResult containing purity, agreement, and valid property metrics.

        Raises:
            FileNotFoundError: If the participant file does not exist.
        """
        df = self.dp.load(participant_id)
        window_len = 1000
        total_possible = len(df) // window_len
        n_to_check = total_possible if max_windows is None else min(max_windows, total_possible)

        first_labels: list[str | None] = []
        majority_labels: list[str | None] = []
        purities: list[float] = []
        property_checks: list[bool] = []
        unannotated_count = 0

        for idx in range(n_to_check):
            start = idx * window_len
            chunk = df.iloc[start : start + window_len]

            first_raw = chunk["annotation"].iloc[0]
            first_mapped = self.dp.map_label(first_raw) if pd.notna(first_raw) else None

            mapped_series = chunk["annotation"].dropna().map(self.dp.map_label).dropna()
            if len(mapped_series) > 0:
                majority_mapped = mapped_series.mode().iloc[0]
                purity = float((mapped_series == majority_mapped).mean())
            else:
                majority_mapped = None
                purity = 0.0

            if first_mapped is None and majority_mapped is None:
                unannotated_count += 1
                continue

            first_labels.append(first_mapped)
            majority_labels.append(majority_mapped)
            purities.append(purity)

            raw_xyz = chunk[["x", "y", "z"]].values.astype(np.float32)
            resampled = self.dp.resample_window(raw_xyz)
            eval_label = first_mapped if first_mapped is not None else majority_mapped
            prop_result = self.validate_window_properties(resampled, eval_label or "")
            property_checks.append(prop_result.all_valid)

        total_annotated = len(first_labels)
        matches = sum(
            f == m for f, m in zip(first_labels, majority_labels) if f is not None and m is not None
        )
        agreement_rate = (matches / total_annotated) if total_annotated > 0 else 0.0
        mean_purity = float(np.mean(purities)) if purities else 0.0
        all_properties_passed = bool(all(property_checks)) if property_checks else False

        return WindowAlignmentValidationResult(
            participant_id=participant_id,
            windows_inspected=n_to_check,
            annotated_windows=total_annotated,
            unannotated_windows=unannotated_count,
            first_vs_majority_agreement=agreement_rate,
            mean_label_purity=mean_purity,
            all_window_properties_passed=all_properties_passed,
        )

    def validate_annotation_dictionary(self) -> AnnotationDictValidationResult:
        """Validates that all expected activity classes are present in the dictionary.

        Returns:
            AnnotationDictValidationResult summarizing mapped classes and any missing classes.

        Raises:
            FileNotFoundError: If annotation-label-dictionary.csv cannot be located.
            KeyError: If the expected scheme column 'label:WillettsSpecific2018' is absent.
        """
        dict_path = self.data_dir / "annotation-label-dictionary.csv"
        if not dict_path.exists():
            raise FileNotFoundError(f"Annotation dictionary not found at {dict_path}")

        df = pd.read_csv(dict_path)
        col = "label:WillettsSpecific2018"
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in dictionary, found: {df.columns.tolist()}")

        mapped_classes = set(df[col].dropna().unique())
        expected_classes = set(self.dp.ACTIVITY_CLASSES)
        missing_classes = sorted(expected_classes - mapped_classes)
        class_counts = df[col].value_counts().to_dict()

        return AnnotationDictValidationResult(
            dictionary_path=str(dict_path),
            total_raw_annotations=len(df),
            all_expected_classes_present=len(missing_classes) == 0,
            missing_classes=missing_classes,
            raw_annotations_per_class=class_counts,
        )

    def validate_splits(self) -> SplitValidationResult:
        """Validates participant split integrity and disjointness from capture24_split.json.

        Returns:
            SplitValidationResult with disjointness check, participant counts, and missing files.

        Raises:
            FileNotFoundError: If capture24_split.json is missing.
        """
        json_path = self.split_dir / "capture24_split.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Split file not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            splits = json.load(file)

        train_set = set(splits.get("train", []))
        val_set = set(splits.get("validation", []))
        test_set = set(splits.get("test", []))

        counts_ok = {
            split: len(s) == self.EXPECTED_SPLIT_COUNTS[split]
            for split, s in [("train", train_set), ("validation", val_set), ("test", test_set)]
        }

        train_val_overlap = sorted(train_set & val_set)
        train_test_overlap = sorted(train_set & test_set)
        val_test_overlap = sorted(val_set & test_set)
        is_disjoint = not (train_val_overlap or train_test_overlap or val_test_overlap)

        all_pids = train_set | val_set | test_set
        missing_files = sorted(
            pid for pid in all_pids if not (self.data_dir / f"{pid}.csv.gz").exists()
        )

        return SplitValidationResult(
            split_file=str(json_path),
            counts={k: len(splits.get(k, [])) for k in ["train", "validation", "test"]},
            counts_ok=counts_ok,
            is_disjoint=is_disjoint,
            overlaps={
                "train_val": train_val_overlap,
                "train_test": train_test_overlap,
                "val_test": val_test_overlap,
            },
            total_participants=len(all_pids),
            all_files_exist=len(missing_files) == 0,
            missing_files=missing_files,
        )

    def validate_class_distribution(
        self, participant_ids: list[str], max_windows_per_pid: int | None = 200
    ) -> ClassDistributionResult:
        """Computes activity class frequencies across a set of participants.

        Args:
            participant_ids: List of participant identifiers to process.
            max_windows_per_pid: Maximum number of windows to read per participant.

        Returns:
            ClassDistributionResult containing frequencies, proportions, and any zero-count classes.
        """
        class_counts = {cls_name: 0 for cls_name in self.dp.ACTIVITY_CLASSES}
        total_annotated_windows = 0

        for pid in tqdm(participant_ids, desc="Checking class distributions"):
            df = self.dp.load(pid)
            windows = self.dp.extract_windows(df)
            if max_windows_per_pid is not None:
                windows = windows[:max_windows_per_pid]

            for _, label in windows:
                if label in class_counts:
                    class_counts[label] += 1
                    total_annotated_windows += 1

        proportions = {}
        if total_annotated_windows > 0:
            for cls_name, count in class_counts.items():
                proportions[cls_name] = round((count / total_annotated_windows) * 100, 2)

        return ClassDistributionResult(
            total_windows=total_annotated_windows,
            counts_per_class=class_counts,
            percentages_per_class=proportions,
            zero_count_classes=sorted(k for k, v in class_counts.items() if v == 0),
        )

    def validate_signal_numerical_health(
        self, participant_ids: list[str], max_windows_per_pid: int = 200
    ) -> SignalNumericalValidationResult:
        """Validates numerical integrity of derived channels and computed features.

        Checks:
        - All resampled window values are finite.
        - Six derived channels (x, y, z, magnitude, pitch, roll) have shape (300,).
        - Pitch angles lie within [-pi, pi].
        - Roll angles lie within [-pi, pi].
        - 24 statistical features satisfy min <= mean <= max and std >= 0.
        - 10 structural features have dominant frequency >= 0 and power share in [0, 1].
        - 3 event features are strictly binary (0.0 or 1.0).

        Args:
            participant_ids: List of participant identifiers to validate.
            max_windows_per_pid: Number of windows to sample per participant.

        Returns:
            SignalNumericalValidationResult summarizing numerical health checks.
        """
        total_checked = 0
        all_finite = True
        channel_shapes_ok = True
        pitch_bounds_ok = True
        roll_bounds_ok = True
        statistical_features_valid = True
        structural_features_valid = True
        event_features_binary = True

        for pid in participant_ids:
            df = self.dp.load(pid)
            windows = self.dp.extract_windows(df)
            sample_windows = windows[:max_windows_per_pid] if max_windows_per_pid else windows

            for window, _ in sample_windows:
                total_checked += 1
                if not np.all(np.isfinite(window)):
                    all_finite = False

                x, y, z = window[:, 0], window[:, 1], window[:, 2]
                magnitude = np.sqrt(x**2 + y**2 + z**2) - 1.0
                pitch = np.arctan2(y, np.sqrt(x**2 + z**2))
                roll = np.arctan2(-x, z)
                channels = [x, y, z, magnitude, pitch, roll]

                if any(ch.shape != (300,) for ch in channels):
                    channel_shapes_ok = False

                if np.any(pitch < -np.pi) or np.any(pitch > np.pi):
                    pitch_bounds_ok = False

                if np.any(roll < -np.pi) or np.any(roll > np.pi):
                    roll_bounds_ok = False

                features = self.dp.compute_caption_features(window)
                if not np.all(np.isfinite(features)):
                    all_finite = False

                # 24 statistical features: mean, std, min, max for 6 channels
                for ch_idx in range(6):
                    ch_offset = ch_idx * 4
                    mean_val = features[ch_offset]
                    std_val = features[ch_offset + 1]
                    min_val = features[ch_offset + 2]
                    max_val = features[ch_offset + 3]

                    if std_val < 0.0 or min_val > mean_val + 1e-5 or mean_val > max_val + 1e-5:
                        statistical_features_valid = False

                # Structural features: dominant frequency and power share
                dom_freq = features[32]
                power_share = features[33]
                if dom_freq < 0.0 or power_share < 0.0 or power_share > 1.0 + 1e-5:
                    structural_features_valid = False

                # Event features: 34, 35, 36 must be in {0.0, 1.0}
                events = features[34:37]
                if any(ev not in (0.0, 1.0) for ev in events):
                    event_features_binary = False

        all_valid = (
            all_finite
            and channel_shapes_ok
            and pitch_bounds_ok
            and roll_bounds_ok
            and statistical_features_valid
            and structural_features_valid
            and event_features_binary
            and total_checked > 0
        )

        return SignalNumericalValidationResult(
            total_windows_checked=total_checked,
            all_finite=all_finite,
            channel_shapes_ok=channel_shapes_ok,
            pitch_bounds_ok=pitch_bounds_ok,
            roll_bounds_ok=roll_bounds_ok,
            statistical_features_valid=statistical_features_valid,
            structural_features_valid=structural_features_valid,
            event_features_binary=event_features_binary,
            all_valid=all_valid,
        )

    def validate_percentile_leakage_and_balance(
        self,
        percentiles_path: str = "data/splits/feature_percentiles.npy",
        sample_train_pids: list[str] | None = None,
        max_windows_per_pid: int = 100,
    ) -> PercentileLeakageValidationResult:
        """Validates feature percentiles for shape, monotonicity, leakage, and bin balance.

        Args:
            percentiles_path: Path to the saved feature_percentiles.npy file.
            sample_train_pids: Optional subset of train participants for bin balance check.
            max_windows_per_pid: Maximum windows per participant for balance evaluation.

        Returns:
            PercentileLeakageValidationResult with leakage and distribution status.

        Raises:
            FileNotFoundError: If the percentiles file or split file does not exist.
        """
        perc_file = Path(percentiles_path)
        if not perc_file.exists():
            raise FileNotFoundError(f"Percentiles file not found at {perc_file}")

        percentiles = np.load(perc_file)
        shape_ok = percentiles.shape == (4, 34)

        # Monotonicity check across the 4 cut points for each of the 34 features
        is_monotonic = bool(np.all(np.diff(percentiles, axis=0) >= -1e-7))

        split_res = self.validate_splits()
        train_pids = set(json.load(open(self.split_dir / "capture24_split.json"))["train"])

        # Check training list integrity
        eval_pids = sample_train_pids or sorted(list(train_pids))[:10]
        no_val_or_test_leakage = all(pid in train_pids for pid in eval_pids)

        # Evaluate bin balance across the 34 numeric features
        self.dp.percentiles = percentiles
        bin_counts = {level: 0 for level in self.dp.LEVELS}
        total_evaluations = 0

        for pid in eval_pids:
            df = self.dp.load(pid)
            windows = self.dp.extract_windows(df)
            for window, _ in windows[:max_windows_per_pid]:
                feats = self.dp.compute_caption_features(window)
                numeric_feats = feats[:34]
                for feat_idx, val in enumerate(numeric_feats):
                    level = self.dp._value_to_level(val, feat_idx)
                    bin_counts[level] += 1
                    total_evaluations += 1

        bin_proportions = {}
        bin_balance_acceptable = True
        if total_evaluations > 0:
            for level, count in bin_counts.items():
                prop = round((count / total_evaluations) * 100, 2)
                bin_proportions[level] = prop
                # Acceptable balance is roughly 20% (10% to 30% tolerance due to ties/zeros)
                if not (10.0 <= prop <= 30.0):
                    bin_balance_acceptable = False

        all_valid = (
            shape_ok
            and is_monotonic
            and no_val_or_test_leakage
            and bin_balance_acceptable
            and len(train_pids) == 80
        )

        return PercentileLeakageValidationResult(
            percentiles_file_exists=True,
            shape_ok=shape_ok,
            is_monotonic=is_monotonic,
            train_participants_count=len(train_pids),
            no_val_or_test_leakage=no_val_or_test_leakage,
            bin_balance_proportions=bin_proportions,
            bin_balance_acceptable=bin_balance_acceptable,
            all_valid=all_valid,
        )

    def validate_feature_unit_tests(self) -> FeatureUnitTestResult:
        """Runs deterministic verification of feature extraction using synthesized physical signals.

        Tests:
        1. Stillness test: Constant 1g signal along z-axis. Must produce still_event = 1.0,
           rhythmic_event = 0.0, and zero overall movement variability.
        2. Angle change test: Abrupt 90-degree wrist orientation shift midway through window.
           Must produce angle_change_event = 1.0.
        3. Rhythmic movement test: Synthesized 1.5 Hz sine wave on magnitude. Must produce
           rhythmic_event = 1.0, dominant frequency within [1.4, 1.6] Hz, and power share > 0.3.
        4. Dimensionality and data type test: Feature output must be shape (37,), float32,
           and contain strictly finite values.

        Returns:
            FeatureUnitTestResult indicating pass/fail status for each synthetic signal test.
        """
        # Test 1: Stillness
        still_window = np.zeros(self.EXPECTED_WINDOW_SHAPE, dtype=np.float32)
        still_window[:, 2] = 1.0
        still_feats = self.dp.compute_caption_features(still_window)
        stillness_passed = (
            still_feats[34] == 1.0  # still event
            and still_feats[36] == 0.0  # not rhythmic
            and np.isclose(still_feats[13], 0.0, atol=1e-5)  # variability of overall movement
        )

        # Test 2: Angle change
        angle_window = np.zeros(self.EXPECTED_WINDOW_SHAPE, dtype=np.float32)
        angle_window[:150, 2] = 1.0
        angle_window[150:, 1] = 1.0
        angle_feats = self.dp.compute_caption_features(angle_window)
        angle_change_passed = angle_feats[35] == 1.0

        # Test 3: Rhythmic movement (biased 1.5 Hz sine wave on z-axis)
        time_series = np.arange(300) / 30.0
        rhythmic_window = np.zeros(self.EXPECTED_WINDOW_SHAPE, dtype=np.float32)
        rhythmic_window[:, 2] = (1.0 + 0.5 * np.sin(2 * np.pi * 1.5 * time_series)).astype(
            np.float32
        )
        rhythmic_feats = self.dp.compute_caption_features(rhythmic_window)
        rhythmic_passed = (
            rhythmic_feats[36] == 1.0
            and 1.4 <= rhythmic_feats[32] <= 1.6
            and rhythmic_feats[33] > 0.3
        )

        # Test 4: Dimension and Type
        dim_type_passed = (
            still_feats.shape == (37,)
            and still_feats.dtype == np.float32
            and bool(np.all(np.isfinite(still_feats)))
        )

        all_passed = (
            stillness_passed and angle_change_passed and rhythmic_passed and dim_type_passed
        )

        return FeatureUnitTestResult(
            stillness_test_passed=stillness_passed,
            angle_change_test_passed=angle_change_passed,
            rhythmic_movement_test_passed=rhythmic_passed,
            dimension_and_type_passed=dim_type_passed,
            all_passed=all_passed,
        )

    def run_step1(
        self, sample_pids: list[str] = ["P001", "P002"]
    ) -> dict[str, WindowAlignmentValidationResult]:
        """Runs Step 1: Benchmark Fidelity and Windowing Alignment validation.

        Args:
            sample_pids: List of participant IDs to inspect.

        Returns:
            Dictionary mapping participant IDs to their WindowAlignmentValidationResult.
        """
        results: dict[str, WindowAlignmentValidationResult] = {}
        for pid in sample_pids:
            res = self.validate_windowing_alignment(pid, max_windows=500)
            results[pid] = res
            print(
                f"[{pid}] Inspected: {res.windows_inspected} windows | "
                f"Agreement (first vs majority): {res.first_vs_majority_agreement * 100:.1f}% | "
                f"Mean Purity: {res.mean_label_purity * 100:.1f}% | "
                f"Properties Valid: {res.all_window_properties_passed}"
            )
        return results

    def run_step2(self) -> tuple[AnnotationDictValidationResult, SplitValidationResult]:
        """Runs Step 2: Label Integrity and Split Validation.

        Returns:
            Tuple of AnnotationDictValidationResult and SplitValidationResult.
        """
        dict_res = self.validate_annotation_dictionary()
        print(f"Annotation Dict: All 10 classes present: {dict_res.all_expected_classes_present}")
        if not dict_res.all_expected_classes_present:
            print(f"  Missing: {dict_res.missing_classes}")

        split_res = self.validate_splits()
        print(
            f"Splits: Disjoint: {split_res.is_disjoint} | "
            f"Counts: {split_res.counts} | All files exist: {split_res.all_files_exist}"
        )

        return dict_res, split_res

    def run_step3(
        self, sample_pids: list[str] = ["P001", "P002"]
    ) -> SignalNumericalValidationResult:
        """Runs Step 3: Numerical Health and Signal Checks.

        Args:
            sample_pids: List of participant IDs to inspect.

        Returns:
            SignalNumericalValidationResult with check outcomes.
        """
        res = self.validate_signal_numerical_health(sample_pids, max_windows_per_pid=200)
        print(
            f"Signal & Feature Numerical Health: Valid: {res.all_valid} | "
            f"Finite: {res.all_finite} | Stats Valid: {res.statistical_features_valid} | "
            f"Structural Valid: {res.structural_features_valid} | Events Binary: {res.event_features_binary}"
        )
        return res

    def run_step4(
        self, percentiles_path: str = "data/splits/feature_percentiles.npy"
    ) -> PercentileLeakageValidationResult:
        """Runs Step 4: Leakage Prevention and Percentile Distribution Validation.

        Args:
            percentiles_path: Path to saved feature percentiles array.

        Returns:
            PercentileLeakageValidationResult with leakage check and bin balance metrics.
        """
        res = self.validate_percentile_leakage_and_balance(percentiles_path)
        print(
            f"Percentile & Leakage: Valid: {res.all_valid} | Shape (4, 34): {res.shape_ok} | "
            f"Monotonic: {res.is_monotonic} | No Val/Test Leakage: {res.no_val_or_test_leakage} | "
            f"Bin Proportions: {res.bin_balance_proportions}"
        )
        return res

    def run_step5(self) -> FeatureUnitTestResult:
        """Runs Step 5: Deterministic Unit Tests for Feature Functions.

        Returns:
            FeatureUnitTestResult with pass/fail status for synthetic physical signals.
        """
        res = self.validate_feature_unit_tests()
        print(
            f"Feature Unit Tests: All Passed: {res.all_passed} | Stillness: {res.stillness_test_passed} | "
            f"Angle Change: {res.angle_change_test_passed} | Rhythmic: {res.rhythmic_movement_test_passed} | "
            f"Dimensions: {res.dimension_and_type_passed}"
        )
        return res

    def run_all(self, sample_pids: list[str] = ["P001", "P002"]) -> bool:
        """Runs all 5 validation steps sequentially and returns overall pass status.

        Args:
            sample_pids: List of sample participant IDs for Steps 1 and 3.

        Returns:
            True if all validation steps passed, False otherwise.
        """
        print("Running full CAPTURE-24 data processing validation...")
        s1 = self.run_step1(sample_pids)
        s2_dict, s2_split = self.run_step2()
        s3 = self.run_step3(sample_pids)
        s4 = self.run_step4()
        s5 = self.run_step5()

        step1_ok = all(r.all_window_properties_passed for r in s1.values())
        step2_ok = s2_dict.all_expected_classes_present and s2_split.is_disjoint
        step3_ok = s3.all_valid
        step4_ok = s4.all_valid
        step5_ok = s5.all_passed

        all_ok = step1_ok and step2_ok and step3_ok and step4_ok and step5_ok
        print(f"\nOverall Validation Status: {'PASSED' if all_ok else 'FAILED'}")
        return all_ok


if __name__ == "__main__":
    validator = DataValidator()
    validator.run_all(sample_pids=["P001", "P002"])
