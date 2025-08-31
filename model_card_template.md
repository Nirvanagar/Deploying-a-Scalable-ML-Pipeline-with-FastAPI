# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Project: Deploying a Machine Learning Pipeline with FastAPI (Udacity)
* Task: Binary classification predicting whether an individual’s income is `>50K` or `<=50K`.
* Algorithm: RandomForestClassifier (`n_estimators=200`, `random_state=42`, `n_jobs=-1`)
* Frameworks & Versions: scikit-learn 1.5.1, pandas 2.2.2, Python 3.10
* Preprocessing: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical features; `LabelBinarizer` for target.
* Artifacts: `model/model.pkl`, `model/encoder.pkl`, `model/lb.pkl`
* Code entry points: `ml/data.py` (`process_data`), `ml/model.py` (training/inference/IO), `train_model.py` (pipeline driver)

## Intended Use
* Primary: Educational demo of an end-to-end ML pipeline (training, evaluation, slicing, and later FastAPI serving).
* Out-of-scope / Not for: Real-world employment, credit, housing, or any high-stakes decisions. This model is not audited for compliance, fairness, or robustness and should not be used to make consequential decisions about people.

## Training Data
* Source: `data/census.csv` (Adult Census Income dataset variant packaged with the starter kit).
* Size & Schema: 32,561 rows × 15 columns.
* Categorical features: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
* Numeric features: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
* Label: `salary` with values `<=50K`, `>50K`.
* Class balance (whole dataset): `>50K` ≈ **7,841** (24%); `<=50K` ≈ **24,720** (76%)
* Split: 80/20 train/test with stratification on `salary`.

## Evaluation Data
* Hold-out set: 20% stratified split from the same CSV.
* Preprocessing for eval: Re-use the fitted one-hot encoder and label binarizer from training (`training=False` with saved `encoder`/`lb`).

## Metrics
The model predicts the positive class for >50K using the default probability threshold (0.5).

Overall (test set):
Precision: 0.7338 | Recall: 0.6365 | F1: 0.6817

Slice Metrics:
Per-category performance is computed for every unique value of each categorical feature and written to `slice_output.txt` (e.g., by `workclass`, `education`, `sex`, etc.). This enables monitoring for disparate performance across subgroups.

## Ethical Considerations
* Bias & Fairness: Features such as `race`, `sex`, `age` are sensitive. Historical and sampling biases may lead to disparate error rates across subgroups. Use the provided slice analysis to check for gaps in precision/recall/F1.
* Staleness & Representativeness: The dataset reflects a specific time period and may not generalize to current labor markets or geographies.
* Appropriate Use: Do not use for real hiring/compensation decisions. If prototyping beyond coursework, incorporate fairness audits, domain review, and legal/compliance checks.

## Caveats and Recommendations
* Imbalance: The positive class (~24%) is under-represented; consider threshold tuning, class weighting, or resampling if optimizing beyond the baseline.
* Explainability: Tree ensembles are not inherently interpretable; add SHAP/feature importance for insight if needed.
* Calibration: If calibrated probabilities are required, fit a calibration layer (e.g., `CalibratedClassifierCV`) on a validation split.
* Monitoring: Track performance drift and subgroup metrics in production-like settings; re-train regularly if data distribution shifts.
* Reproducibility: Keep library versions pinned (as in `requirements.txt`); artifacts are saved under `model/` for consistent inference.
