import os
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm




location = pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/04-temporal-location.csv')
location["timestamp"] = pd.to_datetime(location["timestamp-second"]).dt.date


features = pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/final-merged-features.csv')

location["timestamp"] = pd.to_datetime(location["timestamp"], errors="coerce").dt.date
features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce").dt.date




# target = 'sis'
# target = 'ohs'
target, location, features = 'rapa', location[location['participant-id'] >= 9], features[features['participant'] >= 9]



# variables = [
#     'community-center',
#     'library',
#     'park',
#     'food-establishment',
#     'religious-place'
# ]


# variables = [
#     'assault-rate',
#     'auto-theft-rate',
#     'bike-theft-rate',
#     'break-and-enter-rate',
#     'homicide-rate',
#     'robbery-rate',
#     'shooting-rate',
#     'theft-from-motor-vehicle-rate',
#     'theft-over-rate',
# ]

path = '/home/ali/PycharmProjects/maison/location/sensor-data/dataset'

census_dir = os.path.join(path, "census")

dfs = []
names = []  # filenames without extensions, aligned with dfs

for fname in os.listdir(census_dir):
    if not fname.lower().endswith(".csv"):
        continue
    dfs.append(pd.read_csv(os.path.join(census_dir, fname)))
    names.append(os.path.splitext(fname)[0])



# characteristic = 'Average after-tax income in 2020 among recipients ($)'
# characteristic = 'Total - Visible minority for the population in private households - 25% sample data'
# characteristic = 'Total - Income statistics in 2020 for the population aged 15 years and over in private households - 25% sample data'
# characteristic = '    English'
# characteristic = '  Condominium'
# characteristic = '  Not condominium'
# characteristic = '  Christian'
# characteristic = '    English'
# characteristic = '    French'
# characteristic = '  Suitable'
characteristic = '  Not suitable'

c = 'C1_COUNT_TOTAL'
# c = 'C1_RATE_TOTAL'

# dfs: list of DataFrames
# names: list of filenames without extension (aligned with dfs)

rows = []

for df, fname in zip(dfs, names):
    try:
        sub = df.loc[df["CHARACTERISTIC_NAME"] == characteristic, c]
        if not sub.isna().all():
            value = float(sub.iloc[0])
            rows.append({
                "variable": value,
                "fname": str(fname)
            })
    except Exception:
        pass

# Final DataFrame
census = pd.DataFrame(rows, columns=["variable", "fname"])
census.rename(columns={"fname": "location-id"}, inplace=True)
print(census.shape)


location[target] = location["timestamp"].map(features.drop_duplicates("timestamp").set_index("timestamp")[target])

for variable in [characteristic]:



    # amenities = pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/05-amenities.csv')
    # location[variable] = location["location-id"].map(amenities.drop_duplicates("location-id").set_index("location-id")[variable])

    # crime =     pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/06-crime-rate.csv')
    # location[variable] = location["location-id"].map(crime.drop_duplicates("location-id").set_index("location-id")[variable])


    location[variable] = location["location-id"].map(census.drop_duplicates("location-id").set_index("location-id")['variable'])



    # location.to_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/location-temp.csv', index=False)



    # df = df[df['participant'] >= 9]
    # features = features[features['participant-id'] >= 9]

    # Change these to your column names
    # columns = ['sis-01', 'sis-02', 'sis-03', 'sis-04', 'sis-05', 'sis-06']
    # columns = ['ohs-01', 'ohs-02', 'ohs-03', 'ohs-04', 'ohs-05', 'ohs-06', 'ohs-07', 'ohs-08', 'ohs-09', 'ohs-10', 'ohs-11', 'ohs-12']
    # columns = ['rapa']

    # df['outcome'] = df[columns].sum(axis=1)
    # df['outcome'] = df[columns]
    #

    # OUTCOME_COL = 'outcome'

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

    # PREDICTOR_COL = 'position-count'
    # PREDICTOR_COL = 'position-duration'
    # PREDICTOR_COL = 'position-distance-travelled'
    # PREDICTOR_COL = 'heartrate-min'


    # for PREDICTOR_COL in _PREDICTOR_COL_:

    # OUTCOME_COL = target
    # PREDICTOR_COL = variable
    #
    # d = location[[OUTCOME_COL, PREDICTOR_COL]].copy()
    # d[OUTCOME_COL] = pd.to_numeric(d[OUTCOME_COL], errors="coerce")
    # d[PREDICTOR_COL] = pd.to_numeric(d[PREDICTOR_COL], errors="coerce")
    # d = d.dropna()
    #
    # y_raw = d[OUTCOME_COL].round().astype(int)
    # levels = np.sort(y_raw.unique())
    # y = y_raw.map({v: i for i, v in enumerate(levels)}).astype(int)
    #
    # x = d[[PREDICTOR_COL]]
    # x = (x - x.mean()) / x.std(ddof=0)
    #
    # model = OrderedModel(y, x, distr="logit")
    # res = model.fit(method="bfgs", disp=False)
    #
    # beta = res.params[PREDICTOR_COL]
    # ci_low, ci_high = res.conf_int().loc[PREDICTOR_COL].tolist()
    #
    # or_val = float(np.exp(beta))
    # or_low = float(np.exp(ci_low))
    # or_high = float(np.exp(ci_high))
    # pval = float(res.pvalues[PREDICTOR_COL])
    #
    # print(f"{or_val:.3f} ({or_low:.3f}–{or_high:.3f}), {pval:.4f}", PREDICTOR_COL)


    # OUTCOME_COL = target
    # PREDICTOR_COL = variable
    #
    # # Keep only needed columns and drop missing rows
    # d = location[[OUTCOME_COL, PREDICTOR_COL]].copy()
    # d[OUTCOME_COL] = pd.to_numeric(d[OUTCOME_COL], errors="coerce")
    # d[PREDICTOR_COL] = pd.to_numeric(d[PREDICTOR_COL], errors="coerce")
    # d = d.dropna()
    #
    # # Outcome as continuous (score 20–30)
    # y = d[OUTCOME_COL].astype(float)
    #
    # # Standardize predictor so beta is per 1 SD increase
    # x = d[[PREDICTOR_COL]].astype(float)
    # x[PREDICTOR_COL] = (x[PREDICTOR_COL] - x[PREDICTOR_COL].mean()) / x[PREDICTOR_COL].std(ddof=0)
    #
    # # Add intercept and fit linear regression
    # X = sm.add_constant(x, has_constant="add")
    # res = sm.OLS(y, X).fit()
    #
    # beta = float(res.params[PREDICTOR_COL])
    # ci_low, ci_high = map(float, res.conf_int().loc[PREDICTOR_COL].tolist())
    # pval = float(res.pvalues[PREDICTOR_COL])
    #
    # print(f"beta={beta:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f}), p={pval:.4f}", PREDICTOR_COL)

    # OUTCOME_COL = target
    # PREDICTOR_COL = variable
    #
    # # Keep only needed columns and drop missing rows
    # d = location[[OUTCOME_COL, PREDICTOR_COL]].copy()
    # d[OUTCOME_COL] = pd.to_numeric(d[OUTCOME_COL], errors="coerce")
    # d[PREDICTOR_COL] = pd.to_numeric(d[PREDICTOR_COL], errors="coerce")
    # d = d.dropna()
    #
    # # Outcome: log-transformed continuous score
    # y = np.log(d[OUTCOME_COL].astype(float))
    #
    # # Standardize predictor so effect is per 1 SD increase
    # x = d[[PREDICTOR_COL]].astype(float)
    # x[PREDICTOR_COL] = (x[PREDICTOR_COL] - x[PREDICTOR_COL].mean()) / x[PREDICTOR_COL].std(ddof=0)
    #
    # # Fit log-linear model
    # X = sm.add_constant(x, has_constant="add")
    # res = sm.OLS(y, X).fit()
    #
    # # Extract multiplicative effect
    # beta = res.params[PREDICTOR_COL]
    # ci_low, ci_high = res.conf_int().loc[PREDICTOR_COL].tolist()
    #
    # or_val = float(np.exp(beta))
    # or_low = float(np.exp(ci_low))
    # or_high = float(np.exp(ci_high))
    # pval = float(res.pvalues[PREDICTOR_COL])
    #
    # print(f"{or_val:.3f} ({or_low:.3f}–{or_high:.3f}), {pval:.4f}", PREDICTOR_COL)


    # ------------------------------------------------------------
    # Inputs (your style)
    # ------------------------------------------------------------
    OUTCOME_COL = target
    PREDICTOR_COL = variable

    # ------------------------------------------------------------
    # 1) Keep only needed columns and drop missing rows
    # ------------------------------------------------------------
    d = location.loc[:, [OUTCOME_COL, PREDICTOR_COL]].copy()
    d[OUTCOME_COL] = pd.to_numeric(d[OUTCOME_COL], errors="coerce")
    d[PREDICTOR_COL] = pd.to_numeric(d[PREDICTOR_COL], errors="coerce")
    d = d.dropna()

    # ------------------------------------------------------------
    # 2) Outcome as ordered categories (keeps ordering)
    #    If your outcome is already integer, you can remove .round()
    # ------------------------------------------------------------
    y_raw = d[OUTCOME_COL].round().astype(int)

    levels = np.sort(y_raw.unique())
    if len(levels) < 3:
        raise ValueError(
            f"{OUTCOME_COL} has only {len(levels)} unique level(s) after cleaning. "
            "Ordinal regression needs multiple ordered categories."
        )

    level_map = {v: i for i, v in enumerate(levels)}
    y = y_raw.map(level_map).astype(int)

    # ------------------------------------------------------------
    # 3) Standardize predictor so OR is per 1 SD increase
    # ------------------------------------------------------------
    x = d[[PREDICTOR_COL]].astype(float).copy()
    sd = x[PREDICTOR_COL].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"{PREDICTOR_COL} has zero (or invalid) standard deviation.")
    x[PREDICTOR_COL] = (x[PREDICTOR_COL] - x[PREDICTOR_COL].mean()) / sd

    # ------------------------------------------------------------
    # 4) Fit ordinal logistic regression (proportional odds, logit link)
    # ------------------------------------------------------------
    model = OrderedModel(endog=y, exog=x, distr="logit")
    res = model.fit(method="bfgs", disp=False)

    # ------------------------------------------------------------
    # 5) Effect size (OR), 95% CI, p-value for predictor
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

    # Optional: full summary
    # print(res.summary())


