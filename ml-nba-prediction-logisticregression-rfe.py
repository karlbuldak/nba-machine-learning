
import pandas as pd

df = pd.read_csv("nba_games.csv", index_col=0)

df

df = df.sort_values("date")

df = df.reset_index(drop=True)

del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]


# +
def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target).reset_index(drop=True)
df = df.copy()

# -

df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")

df["won"].value_counts()

df["target"].value_counts()

nulls = pd.isnull(df).sum()

nulls = nulls[nulls > 0]

valid_columns = df.columns[~df.columns.isin(nulls.index)]

valid_columns

df = df[valid_columns].copy()

df

# +
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline

logistic_model = LogisticRegression(max_iter=1000, verbose=True)

split = TimeSeriesSplit(n_splits=3)

search_space = {
    "classification__penalty": ["l1", "l2"],
    "classification__C": [0.1, 1, 10],
    "classification__solver": ["liblinear", "lbfgs"]
}

rfe = RFE(estimator=logistic_model, 
          n_features_to_select=30
         )

pipeline = Pipeline(steps=[('feature_selection', rfe), ('classification', logistic_model)])

gs = GridSearchCV(estimator=pipeline, param_grid=search_space, cv=split, scoring='accuracy')

# -

removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]

# +
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])
# -

df

gs.fit(df[selected_columns], df["target"])

# +
# Najlepszy model po GridSearchCV
best_model = gs.best_estimator_

# Wyświetlanie najlepszych parametrów
print("Najlepsze parametry: ", gs.best_params_)

best_rfe = best_model.named_steps['feature_selection']
# -

predictors = list(selected_columns[best_rfe.get_support()])

predictors


def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)


predictions = backtest(df, best_model, predictors)

# +
from sklearn.metrics import accuracy_score

accuracy_score(predictions["actual"], predictions["prediction"])
# -

df

# +
df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

def find_team_averages(team):
    numeric_cols = team.select_dtypes(include="number")
    rolling = numeric_cols.rolling(10).mean()
    
    rolling[["team", "season"]] = team[["team", "season"]]
    return rolling

df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

# -

df_rolling

rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols
df = pd.concat([df, df_rolling], axis=1)

df = df.dropna()

df


# +
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")
# -

df

full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])

full

full[["team_x", "team_opp_next_x", "team_y", "team_opp_next_y", "date_next"]]

removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

removed_columns

selected_columns = full.columns[~full.columns.isin(removed_columns)]
rfe.fit(full[selected_columns], full["target"])

predictors = list(selected_columns[rfe.get_support()])

predictors

predictions = backtest(full, logistic_model, predictors)

wynik = accuracy_score(predictions["actual"], predictions["prediction"])
print(f"wynik to {wynik*100:.4f}"+"% dokładności osiągniętej przez Recursive Feature Elimination z najlepszym modelem.")
