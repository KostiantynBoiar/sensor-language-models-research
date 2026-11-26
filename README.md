# Sensor-Language Model Research Experiments

Project checklist for **Do Sensor-Language Models Learn More Than Their Caption Rules?**

## 1. Project setup
- [ ] Set up the repository structure.
- [ ] Set up the Python environment and install dependencies.
- [ ] Configure Git and `.gitignore`.

## 2. Dataset preparation
- [ ] Download and organise CAPTURE-24.
- [ ] Inspect the dataset and annotation files.
- [ ] Create the train, validation and test participant splits.
- [ ] Verify that the splits are participant-disjoint.
- [ ] Save and document the splits.

## 3. Data processing
- [ ] Implement sensor data loading.
- [ ] Implement 10-second window extraction.
- [ ] Implement the 37 caption features.
- [ ] Validate the processed data.

## 4. Caption generation
- [ ] Define the caption statements and templates.
- [ ] Implement caption generation.
- [ ] Ensure captions contain no activity names or time-of-day information.
- [ ] Audit the generated captions.

## 5. Baselines
- [ ] Implement the feature-based rule baseline.
- [ ] Implement the required classification readouts.
- [ ] Establish baseline results.

## 6. Model development
- [ ] Implement the sensor encoder.
- [ ] Integrate the frozen CLIP text encoder.
- [ ] Implement contrastive training.
- [ ] Implement the alternative training controls.
- [ ] Configure experiment seeds and training settings.

## 7. Training and validation
- [ ] Run initial experiments.
- [ ] Tune model settings using validation participants.
- [ ] Compare training controls.
- [ ] Finalise the experimental configuration.

## 8. Evaluation
- [ ] Freeze the experimental protocol.
- [ ] Finalise the statistical analysis plan.
- [ ] Evaluate models on held-out test participants.
- [ ] Compare macro-F1 results.
- [ ] Analyse results and draw conclusions.

## 9. Further experiments
- [ ] Audit existing sensor-language models.
- [ ] Experiment with alternative caption pipelines.
- [ ] Develop and evaluate a simplified model.
- [ ] Compare model performance and computational cost.

## 10. Finalisation
- [ ] Organise experiment results.
- [ ] Document the implementation.
- [ ] Prepare figures and tables.
- [ ] Write up the findings.
- [ ] Clean up and prepare the repository for release.