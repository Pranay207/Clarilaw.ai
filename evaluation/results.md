# Model Evaluation Results

## Dataset
Current evaluation is based on `data/processed/cleaned_legal_texts.csv` with these six categories:

- Bail
- Court Procedure
- FIR Procedure
- General Procedure
- Police Procedure
- Trial Procedure

## Latest Classical ML Run

### Logistic Regression
- Accuracy: 0.6636
- Strong on `General Procedure`
- Weak recall on minority classes such as `Bail` and `FIR Procedure`

### Random Forest
- Accuracy: 0.6822
- Best-performing classical model in the current training script
- Better balance than logistic regression, but still misses some minority classes

## Deployment Note
The training script now saves the best classical model to `models/intent_classifier.pkl`, which is the artifact used by the API.

## Transformer Pipeline
The transformer data preparation step now uses the same `category` labels as the classical pipeline.
Transformer metrics are not recorded here because they depend on downloading the base model during training.

## Summary
- The current dataset is imbalanced, with `General Procedure` dominating the samples.
- Random Forest is the strongest classical baseline for the checked-in data.
- Minority-class performance still needs improvement through balancing, augmentation, or better label coverage.
