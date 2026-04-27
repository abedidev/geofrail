import os
import pandas as pd
import numpy as np
# from pyarrow.conftest import groups
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from tabpfn import TabPFNRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline


root = '/home/ali/PycharmProjects/maison/location/sensor-data'

df = pd.read_csv(os.path.join(root, 'final-merged-features.csv'))
features = pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/02-sensor-features.csv')




path = '/home/ali/PycharmProjects/maison/location/sensor-data/dataset/'
pcodes = pd.read_csv(os.path.join(path, '01-demographics.csv'))
pcodes = pcodes[['participant-id', 'location-id']]

amenities = pd.read_csv(os.path.join(path, '05-amenities.csv'))
crime = pd.read_csv(os.path.join(path, '06-crime-rate.csv'))


features = features.merge(pcodes, on="participant-id", how="left")
features = features.merge(amenities, on="location-id", how="left")
features = features.merge(crime, on="location-id", how="left")









features, df = features[features['participant-id'] >= 9], df[df['participant'] >= 9]

# Change these to your column names
# columns = ['sis-01', 'sis-02', 'sis-03', 'sis-04', 'sis-05', 'sis-06']
# columns = ['ohs-01', 'ohs-02', 'ohs-03', 'ohs-04', 'ohs-05', 'ohs-06', 'ohs-07', 'ohs-08', 'ohs-09', 'ohs-10', 'ohs-11', 'ohs-12']
columns = ['rapa']
# columns = ['cfs']

# df['outcome'] = df[columns].sum(axis=1)
df['outcome'] = df[columns]


OUTCOME_COL = 'outcome'

y = df['outcome'].to_numpy()


feature_names = [


    # 'acceleration-count',
    #    'acceleration-mean', 'acceleration-std', 'acceleration-sum',
    #    'acceleration-entropy', 'acceleration-kurtosis', 'acceleration-skew',
    #    'acceleration-coefficient-of-variation',
    #    'acceleration-minutes-with-data', 'acceleration-hours-with-data',
    #    'acceleration-movement-events-00to06',
    #    'acceleration-movement-events-06to12',
    #    'acceleration-movement-events-12to18',
    #    'acceleration-movement-events-18to24',
    #    'acceleration-movement-events-24h',
    #    'acceleration-intradaily-variability', 'heartrate-count',
    #    'heartrate-min', 'heartrate-max', 'heartrate-mean', 'heartrate-std',
    #    'heartrate-hours-with-data', 'motion-count', 'motion-ratio',
    #    'motion-mean', 'motion-max', 'motion-max-timestamp', 'position-count',
    #    'position-duration', 'position-distance-travelled', 'sleep-total',
    #    'sleep-deep', 'sleep-light', 'sleep-rem', 'sleep-snoring-duration',
    #    'sleep-duration-to-sleep', 'sleep-duration-to-wakeup',
    #    'sleep-wakeup-count', 'sleep-heartrate-mean', 'sleep-heartrate-min',
    #    'sleep-heartrate-max', 'step-count', 'step-ratio', 'step-mean',
    #    'step-max', 'step-max-timestamp',


              'community-center',
              'library',
              'park',
              'food-establishment',
              'religious-place',

              'assault-rate',
              'auto-theft-rate',
              'bike-theft-rate',
              'break-and-enter-rate',
              'homicide-rate',
              'robbery-rate',
              'shooting-rate',
              'theft-from-motor-vehicle-rate',
              'theft-over-rate'

              ]

x = features[feature_names].to_numpy()

p = features['participant-id'].to_numpy()
groups = p



mae_scores = []
r2_scores = []



# # loo = LeaveOneOut()
# # for fold, (train_idx, test_idx) in enumerate(loo.split(x, y), start=1):
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
#
# # logo = LeaveOneGroupOut()
# # for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, groups=p), start=1):
#
#
#
#
#     X_train, X_test = x[train_idx], x[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#
#     # model = CatBoostRegressor(
#     #     iterations=500,
#     #     learning_rate=0.05,
#     #     depth=6,
#     #     loss_function="MAE",
#     #     verbose=False,
#     #     random_seed=42
#     # )
#     # model = TabPFNRegressor(device="cuda", random_state=42)
#     model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
#
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     mae_scores.append(mae)
#     r2_scores.append(r2)
#
#     print(f"Fold {fold}: MAE = {mae:.4f}, R2 = {r2:.4f}")
#
# mae_scores = np.array(mae_scores)
# r2_scores = np.array(r2_scores)
#
# print("\nSummary")
# # print(f"MAE mean ± std: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
# # print(f"R2 mean ± std: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
#
# print(f"{mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
# # print(f"{r2_scores.mean():.4f} ± {r2_scores.std():.4f}")


# x: (n_samples, n_features) numpy array
# y: (n_samples,) numpy array
# feature_names: list of length n_features (optional, for printing)
# Assumes you already imported: numpy as np
# and from sklearn.model_selection import KFold
# and from sklearn.metrics import mean_absolute_error, r2_score
# and from sklearn.svm import SVR
# and from sklearn.preprocessing import StandardScaler
# and from sklearn.feature_selection import RFE
# and from sklearn.pipeline import Pipeline


# Required pre-defined variables:
# model_name in {"svr", "catboost", "tabpfn"}
# lopo: bool
# groups: participant ids (len == len(y))
# feature_names: list of feature names (len == x.shape[1]) OR None


model_name = 'catboost'
lopo = True


model_name = model_name.lower().strip()

if lopo:
    outer_splitter = LeaveOneGroupOut()
    outer_splits = outer_splitter.split(x, y, groups=groups)
else:
    outer_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = outer_splitter.split(x)

mae_scores, r2_scores = [], []
selected_feature_indices_per_fold = []

for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):

    X_train_full, X_test = x[train_idx], x[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    if lopo:
        inner_splitter = LeaveOneGroupOut()
        inner_groups = groups[train_idx]
    else:
        inner_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    n_features_total = X_train_full.shape[1]
    candidate_k = sorted(set([5, 10, 20, 30, 50, n_features_total]))
    candidate_k = [k for k in candidate_k if 1 <= k <= n_features_total]

    best_k, best_inner_mae = None, None

    for k in candidate_k:
        inner_maes = []

        if lopo:
            inner_splits = inner_splitter.split(
                X_train_full, y_train_full, groups=inner_groups
            )
        else:
            inner_splits = inner_splitter.split(X_train_full)

        for tr_i, va_i in inner_splits:
            X_tr, X_va = X_train_full[tr_i], X_train_full[va_i]
            y_tr, y_va = y_train_full[tr_i], y_train_full[va_i]

            if model_name == "svr":
                predictor = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))
                ])
            elif model_name == "catboost":
                predictor = CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="MAE",
                    verbose=False,
                    random_seed=42
                )
            elif model_name == "tabpfn":
                predictor = TabPFNRegressor(device="cuda", random_state=42)
            else:
                raise ValueError("model_name must be 'svr', 'catboost', or 'tabpfn'")

            pipe = Pipeline([
                ("scale_for_rfe", StandardScaler()),
                ("rfe", RFE(
                    estimator=SVR(kernel="linear", C=1.0),
                    n_features_to_select=k,
                    step=0.1
                )),
                ("predictor", predictor)
            ])

            pipe.fit(X_tr, y_tr)
            y_va_pred = pipe.predict(X_va)
            inner_maes.append(mean_absolute_error(y_va, y_va_pred))

        mean_inner_mae = float(np.mean(inner_maes))
        if (best_inner_mae is None) or (mean_inner_mae < best_inner_mae):
            best_inner_mae = mean_inner_mae
            best_k = k

    # -------- Final fit on outer training split --------
    if model_name == "svr":
        final_predictor = Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))
        ])
    elif model_name == "catboost":
        final_predictor = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            verbose=False,
            random_seed=42
        )
    elif model_name == "tabpfn":
        final_predictor = TabPFNRegressor(device="cuda", random_state=42)

    final_pipe = Pipeline([
        ("scale_for_rfe", StandardScaler()),
        ("rfe", RFE(
            estimator=SVR(kernel="linear", C=1.0),
            n_features_to_select=best_k,
            step=0.1
        )),
        ("predictor", final_predictor)
    ])

    final_pipe.fit(X_train_full, y_train_full)
    y_pred = final_pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae_scores.append(mae)
    r2_scores.append(r2)

    support_mask = final_pipe.named_steps["rfe"].support_
    selected_idx = np.where(support_mask)[0]
    selected_feature_indices_per_fold.append(selected_idx)

    n_selected = len(selected_idx)

    print(f"\nFold {fold} | {model_name} | k={best_k} | selected={n_selected}")

    if feature_names is not None:
        print("Selected features:")
        for f in [feature_names[i] for i in selected_idx]:
            print(f"  - {f}")
    else:
        print(f"Selected feature indices: {selected_idx.tolist()}")

    print(f"MAE={mae:.4f} | R2={r2:.4f}")

mae_scores = np.array(mae_scores, dtype=float)
r2_scores = np.array(r2_scores, dtype=float)

print("\nSummary")
print(f"{model_name} | MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
print(f"{model_name} | R2 : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
