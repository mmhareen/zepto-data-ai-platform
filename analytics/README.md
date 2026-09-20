# Analytics Pipeline — Module 2

End-to-end EDA and predictive modeling on the Titanic dataset: profiling,
cleaning, visual storytelling, then a full classification + regression
modeling pipeline with rigorous train/test hygiene.

## Setup

From the repo root:
\`\`\`
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## How to run

Run in order from the repo root (with the venv active):

\`\`\`
python analytics\01_load_and_profile.py    # loads titanic dataset, cleans it, saves titanic.csv
python analytics\02_univariate.py          # age/fare histograms, box plots, IQR outliers
python analytics\03_bivariate.py           # survival rates by sex/pclass, correlation heatmap
python analytics\04_data_story.py          # 4-chart multivariate story
python analytics\05_standardization_check.py  # z-score sanity check (exploratory only)
python analytics\06_modeling.py            # full modeling pipeline (classifiers, tuning, regression, joblib save)
\`\`\`

The raw dataset is loaded from `sns.load_dataset("titanic")` exactly once,
in `01_load_and_profile.py`. Every later script reads from the committed
`titanic.csv` this script produces, so the whole module can be graded
offline after the first run.

## Part A — EDA & Cleaning

### Missing values
| Column | % Missing | Strategy |
|---|---|---|
| deck | 77.22% | Dropped column entirely — too high to impute reliably |
| age | 19.87% | Median imputation (5–30% threshold) |
| embarked | 0.22% | Rows dropped (<5% threshold) |
| embark_town | 0.22% | Rows dropped (<5% threshold) |

No imputation was attempted for `deck` since over 3/4 of its values were
missing — filling that much data would be mostly fabrication rather than
genuine signal, and the column isn't used in modeling anyway.

### Univariate analysis (age & fare)
- **Outliers (IQR rule):** age has 65 outliers (bounds: 2.50–54.50); fare
  has 114 outliers (bounds: -26.76–65.66 — the negative lower bound has no
  real meaning since fares can't be negative, so all fare outliers are on
  the high end).
- **Fare skewness:** mode (£8.05) < median (£14.45) < mean (£32.10) —
  confirms fare is right-skewed. A small number of very high fares (up to
  ~£512) pull the mean well above the median/mode.

### Bivariate analysis
- Survival rate by sex: male 18.89%, female 74.04%
- Survival rate by pclass: 1st 62.62%, 2nd 47.28%, 3rd 24.24%
- Survival rate by sex+pclass: female 1st 96.74%, female 2nd 92.11%,
  female 3rd 50.00%, male 1st 36.89%, male 2nd 15.74%, male 3rd 13.54%
- **Correlation matrix** (6 numeric columns: survived, pclass, age, sibsp,
  parch, fare — `adult_male`/`alone` excluded as derived/redundant flags):
  the two strongest correlations are **pclass↔fare (-0.55)** — lower class
  number (better class) means higher fare, as expected — and
  **sibsp↔parch (0.41)** — both are proxies for "traveling with family."

### Multivariate data story (4 charts)
1. **Survival by sex** — women survived ~4x more often than men (74% vs 19%).
2. **Survival by pclass** — survival drops steadily from 1st (63%) to 3rd (24%) class.
3. **Survival by class+sex** — sex mattered within every class; even a
   3rd-class woman (50%) outsurvived a 1st-class man (37%).
4. **Age by survival outcome** — nearly identical age distributions between
   survivors and non-survivors; age alone doesn't clearly separate the two
   groups in this dataset.

### Standardization check (exploratory only, not used in modeling)
Before: age mean 29.32/std 12.98, fare mean 32.10/std 49.70.
After z-score: both columns confirmed mean ≈ 0, std ≈ 1.

## Part B — Modeling

### Train/test split
80/20 stratified split (`stratify=y`) — justified by class imbalance
(~38% survived). Confirmed survival rates: full 38.25%, train 38.26%,
test 38.2%.

### Preprocessing
`ColumnTransformer` + `Pipeline`: numeric columns (pclass, age, sibsp,
parch, fare) median-imputed + standard-scaled; categorical columns (sex,
embarked) mode-imputed + one-hot encoded. All fitting happens only on the
training split; the test split only ever gets `.transform()`.

### Classifier comparison
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7697 | 0.6901 | 0.7206 | 0.7050 | 0.7541 |
| Random Forest | 0.8202 | 0.7812 | 0.7353 | 0.7576 | 0.8179 |

The Decision Tree overfit clearly (98.45% train vs. 76.97% test accuracy);
Random Forest's ensemble averaging reduced this gap substantially
(98.45% train vs. 82.02% test) and delivered the best overall test
performance.

### Imbalance handling comparison (Random Forest)
| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7812 | 0.7353 | 0.7576 |
| class_weight='balanced' | 0.7391 | 0.7500 | 0.7445 |
| SMOTE (train fold only) | 0.7460 | 0.6912 | 0.7176 |

**Conclusion:** the baseline (no handling) performed best on F1. This
dataset's imbalance (62/38) is mild enough that neither reweighting nor
synthetic oversampling improved results — a reminder that imbalance
techniques should be validated on the actual data, not applied by default.

### Hyperparameter tuning (GridSearchCV on Random Forest)
Best parameters: `max_depth=15, max_features='sqrt', n_estimators=200`.
Best CV F1: 0.7471. OOB score: 0.8073. Test accuracy: 0.8146 — essentially
unchanged from the untuned baseline (0.8202), so tuning offered negligible
gains on this dataset.

### Regression side-task (predicting fare)
Features: pclass, age, sibsp, parch, sex, embarked (fare and survived
excluded as target/other-task's-target respectively).
- MAE: 21.14, RMSE: 41.75, R²: 0.3468, Adjusted R²: 0.3239
- RMSE nearly double MAE indicates a few large errors are inflating it —
  consistent with fare's extreme right-skew.
- **Residual plot shows clear heteroscedasticity**: residuals stay tight
  near zero for low predicted fares but fan out dramatically (up to +430)
  for higher predicted fares, since a linear model struggles most on the
  small number of very expensive fares.

### Final recommendation
**Deploy the baseline Random Forest classifier.** It has the best accuracy
(0.8202), F1 (0.7576), and recall (0.7353) of the three classifiers.
Logistic Regression's higher AUC (0.8610) reflects better ranking across
all thresholds, but F1/accuracy at the standard threshold matter more for
real deployed yes/no predictions. The Decision Tree is not recommended due
to clear overfitting. Since GridSearchCV tuning and imbalance handling
both failed to meaningfully improve on the untuned baseline, the simplest
version — baseline Random Forest, no extra handling — is the practical
choice.

### Saved artifact
The complete fitted pipeline (preprocessing + Random Forest) is saved to
`analytics/best_model_pipeline.joblib` via `joblib.dump`, and verified to
reload and predict correctly on raw, unpreprocessed test data.

## Files
- `01_load_and_profile.py` … `05_standardization_check.py` — EDA pipeline
- `06_modeling.py` — full modeling pipeline
- `titanic.csv` — cleaned data, offline fallback
- `*.png` — saved charts
- `best_model_pipeline.joblib` — deployable saved model