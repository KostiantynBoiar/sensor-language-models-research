
import csv
import json
import random
from pathlib import Path


class Capture24Splitter:
    def __init__(self, data_dir, output_dir, seed=42):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed

    def get_participants(self):
        return sorted(
            int(path.name.removesuffix(".csv.gz")[1:])
            for path in self.data_dir.glob("P*.csv.gz")
            if path.name.removesuffix(".csv.gz")[1:].isdigit()
        )

    def create_split(self):
        development = list(range(1, 101))
        test = list(range(101, 152))

        rng = random.Random(self.seed)
        validation = sorted(rng.sample(development, 20))
        train = sorted(set(development) - set(validation))

        return {
            "seed": self.seed,
            "train": [f"P{i:03d}" for i in train],
            "validation": [f"P{i:03d}" for i in validation],
            "test": [f"P{i:03d}" for i in test],
        }

    def save(self, splits):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with (self.output_dir / "capture24_split.json").open("w") as file:
            json.dump(splits, file, indent=4)

        with (self.output_dir / "capture24_split.csv").open(
            "w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["participant", "split"])

            for split_name in ("train", "validation", "test"):
                for participant in splits[split_name]:
                    writer.writerow([participant, split_name])

    def run(self):
        participants = self.get_participants()

        if len(participants) != 151:
            raise ValueError(
                f"Expected 151 participants, found {len(participants)}."
            )

        splits = self.create_split()
        self.save(splits)

        print(f"Training:   {len(splits['train'])}")
        print(f"Validation: {len(splits['validation'])}")
        print(f"Test:       {len(splits['test'])}")
        print(f"Saved to:   {self.output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    splitter = Capture24Splitter(
        data_dir=project_root / "data" / "capture24",
        output_dir=project_root / "data" / "splits",
        seed=42,
    )

    splitter.run()
