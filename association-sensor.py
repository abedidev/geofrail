import os
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

root = '/home/ali/PycharmProjects/maison/location/sensor-data'

df = pd.read_csv(os.path.join(root, 'final-merged-features.csv'))
features = pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/02-sensor-features.csv')

# df, features = df[df['participant'] >= 9], features[features['participant-id'] >= 9]

# Change these to your column names
columns = ['sis-01', 'sis-02', 'sis-03', 'sis-04', 'sis-05', 'sis-06']
# columns = ['ohs-01', 'ohs-02', 'ohs-03', 'ohs-04', 'ohs-05', 'ohs-06', 'ohs-07', 'ohs-08', 'ohs-09', 'ohs-10', 'ohs-11', 'ohs-12']
# columns = ['rapa']

df['outcome'] = df[columns].sum(axis=1)
# df['outcome'] = df[columns]


OUTCOME_COL = 'outcome'

# _PREDICTOR_COL_ =[
# 'acceleration-count',
# 'acceleration-mean',
# 'acceleration-std',
# 'acceleration-sum',
# 'acceleration-entropy',
# 'acceleration-kurtosis',
# 'acceleration-skew',
# 'acceleration-coefficient-of-variation',
# 'acceleration-minutes-with-data',
# 'acceleration-hours-with-data',
# 'acceleration-movement-events-00to06',
# 'acceleration-movement-events-06to12',
# 'acceleration-movement-events-12to18',
# 'acceleration-movement-events-18to24',
# 'acceleration-movement-events-24h',
# 'acceleration-intradaily-variability',
# ]

# _PREDICTOR_COL_ =[
# 'heartrate-count',
# 'heartrate-min',
# 'heartrate-max',
# 'heartrate-mean',
# 'heartrate-std',
# 'heartrate-hours-with-data',
# ]

# _PREDICTOR_COL_ =[
# 'motion-count',
# 'motion-ratio',
# 'motion-mean',
# 'motion-max',
# 'motion-max-timestamp',
# ]

# _PREDICTOR_COL_ =[
# 'sleep-total',
# 'sleep-deep',
# 'sleep-light',
# 'sleep-rem',
# 'sleep-snoring-duration',
# 'sleep-duration-to-sleep',
# 'sleep-duration-to-wakeup',
# 'sleep-wakeup-count',
# 'sleep-heartrate-mean',
# 'sleep-heartrate-min',
# 'sleep-heartrate-max',
# ]

# _PREDICTOR_COL_ =[
# 'step-count',
# 'step-ratio',
# 'step-mean',
# 'step-max',
# 'step-max-timestamp'
# ]

_PREDICTOR_COL_ = [
    'position-count',
    'position-duration',
    'position-distance-travelled'
]

for PREDICTOR_COL in _PREDICTOR_COL_:

    # Inputs you already have
    OUTCOME = df[OUTCOME_COL]  # pandas Series
    PREDICTOR = features[PREDICTOR_COL]  # pandas Series

    # ------------------------------------------------------------
    # 1) Build analysis frame with explicit column names
    # ------------------------------------------------------------
    d = pd.concat(
        [OUTCOME.rename(OUTCOME_COL), PREDICTOR.rename(PREDICTOR_COL)],
        axis=1
    ).copy()

    # Numeric coercion + complete-case
    d[OUTCOME_COL] = pd.to_numeric(d[OUTCOME_COL], errors="coerce")
    d[PREDICTOR_COL] = pd.to_numeric(d[PREDICTOR_COL], errors="coerce")
    d = d.dropna()

    # ------------------------------------------------------------
    # 2) Outcome as ordered categories for proportional-odds model
    #    This operationalizes: logit(P(Y >= k)) = alpha_k + beta * X
    # ------------------------------------------------------------
    y_raw = d[OUTCOME_COL].round().astype(int)  # drop .round() if already integer

    levels = np.sort(y_raw.unique())
    if len(levels) < 3:
        raise ValueError(
            f"{OUTCOME_COL} has only {len(levels)} unique level(s) after cleaning; "
            "ordinal logistic regression needs multiple ordered categories."
        )

    level_map = {v: i for i, v in enumerate(levels)}
    y = y_raw.map(level_map).astype(int)

    # ------------------------------------------------------------
    # 3) Predictor standardized so exp(beta) is OR per +1 SD increase
    # ------------------------------------------------------------
    x = d[[PREDICTOR_COL]].astype(float).copy()
    sd = x[PREDICTOR_COL].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"{PREDICTOR_COL} has zero (or invalid) standard deviation.")
    x[PREDICTOR_COL] = (x[PREDICTOR_COL] - x[PREDICTOR_COL].mean()) / sd

    # ------------------------------------------------------------
    # 4) Fit proportional-odds ordinal logistic regression (logit link)
    # ------------------------------------------------------------
    model = OrderedModel(endog=y, exog=x, distr="logit")
    res = model.fit(method="bfgs", disp=False)

    # ------------------------------------------------------------
    # 5) Effect size: OR, 95% CI, p-value for predictor slope (beta)
    # ------------------------------------------------------------
    beta = float(res.params[PREDICTOR_COL])
    se = float(res.bse[PREDICTOR_COL])
    pval = float(res.pvalues[PREDICTOR_COL])

    z = 1.96
    ci_low = beta - z * se
    ci_high = beta + z * se

    or_val = float(np.exp(beta))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    print(f"{or_val:.3f} ({or_low:.3f}–{or_high:.3f}), {pval:.4f}", PREDICTOR_COL)

