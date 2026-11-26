Updated checklist with what we’ve completed so far:

# Sensor-Language Model Research Experiments

Project checklist for **Do Sensor-Language Models Learn More Than Their Caption Rules?**

## 1. Project setup
- [x] Set up the repository structure.
- [x] Set up the Python environment and install dependencies.
- [x] Configure Git and `.gitignore`.

## 2. Dataset preparation
- [x] Download and organise CAPTURE-24.
- [x] Inspect the dataset and annotation files.
- [x] Create the train, validation and test participant splits.
- [x] Verify that the splits are participant-disjoint.
- [x] Save and document the splits.

## 3. Data processing
- [x] Implement sensor data loading.
- [x] Implement 10-second window extraction.
- [x] Implement the 37 caption features.
- [ ] Validate the processed data.

## 4. Caption generation
- [x] Define the caption statements and templates.
- [x] Implement caption generation.
- [x] Ensure captions contain no activity names or time-of-day information.
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

**Current status**: Sections 1–4 are largely done (except validation/audit). Ready to move to the rule baseline (Section 5) or finish the caption audit.