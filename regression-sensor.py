import os
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from tabpfn import TabPFNRegressor
from sklearn.svm import SVR

root = '/home/ali/PycharmProjects/maison/location/sensor-data'

df = pd.read_csv(os.path.join(root, 'final-merged-features.csv'))
features = pd.read_csv('/home/ali/PycharmProjects/maison/location/sensor-data/dataset/02-sensor-features.csv')

df = df[df['participant'] >= 9]
features = features[features['participant-id'] >= 9]

# Change these to your column names
# columns = ['sis-01', 'sis-02', 'sis-03', 'sis-04', 'sis-05', 'sis-06']
# columns = ['ohs-01', 'ohs-02', 'ohs-03', 'ohs-04', 'ohs-05', 'ohs-06', 'ohs-07', 'ohs-08', 'ohs-09', 'ohs-10', 'ohs-11', 'ohs-12']
columns = ['rapa']
# columns = ['cfs']

# df['outcome'] = df[columns].sum(axis=1)
df['outcome'] = df[columns]


OUTCOME_COL = 'outcome'

y = df['outcome'].to_numpy()

x = features[['acceleration-count',
       'acceleration-mean', 'acceleration-std', 'acceleration-sum',
       'acceleration-entropy', 'acceleration-kurtosis', 'acceleration-skew',
       'acceleration-coefficient-of-variation',
       'acceleration-minutes-with-data', 'acceleration-hours-with-data',
       'acceleration-movement-events-00to06',
       'acceleration-movement-events-06to12',
       'acceleration-movement-events-12to18',
       'acceleration-movement-events-18to24',
       'acceleration-movement-events-24h',
       'acceleration-intradaily-variability', 'heartrate-count',
       'heartrate-min', 'heartrate-max', 'heartrate-mean', 'heartrate-std',
       'heartrate-hours-with-data', 'motion-count', 'motion-ratio',
       'motion-mean', 'motion-max', 'motion-max-timestamp', 'position-count',
       'position-duration', 'position-distance-travelled', 'sleep-total',
       'sleep-deep', 'sleep-light', 'sleep-rem', 'sleep-snoring-duration',
       'sleep-duration-to-sleep', 'sleep-duration-to-wakeup',
       'sleep-wakeup-count', 'sleep-heartrate-mean', 'sleep-heartrate-min',
       'sleep-heartrate-max', 'step-count', 'step-ratio', 'step-mean',
       'step-max', 'step-max-timestamp']].to_numpy()

p = features['participant-id'].to_numpy()


mae_scores = []
r2_scores = []



# loo = LeaveOneOut()
# for fold, (train_idx, test_idx) in enumerate(loo.split(x, y), start=1):

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):

# logo = LeaveOneGroupOut()
# for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, groups=p), start=1):




    X_train, X_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # model = CatBoostRegressor(
    #     iterations=500,
    #     learning_rate=0.05,
    #     depth=6,
    #     loss_function="MAE",
    #     verbose=False,
    #     random_seed=42
    # )
    # model = TabPFNRegressor(device="cuda", random_state=42)
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae_scores.append(mae)
    r2_scores.append(r2)

    print(f"Fold {fold}: MAE = {mae:.4f}, R2 = {r2:.4f}")

mae_scores = np.array(mae_scores)
r2_scores = np.array(r2_scores)

print("\nSummary")
# print(f"MAE mean ± std: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
# print(f"R2 mean ± std: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

print(f"{mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
# print(f"{r2_scores.mean():.4f} ± {r2_scores.std():.4f}")








