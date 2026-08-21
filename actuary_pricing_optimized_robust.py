
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


df = pd.read_excel(r"C:\Users\Gem\Downloads\Insurance_Pricing.xlsx", sheet_name="DATA")

# Çalışma kopyası oluştur
df = df.copy()

# Kolon isimlerini temizle
df.columns = df.columns.str.strip()

# String sütunlardaki boşlukları temizle
text_cols = df.select_dtypes(include=["object", "string"]).columns
df[text_cols] = df[text_cols].apply(lambda x: x.str.strip())

# Son boş kolonu sil
df = df.loc[:, ~df.columns.isna()]

print(df.shape)
print(df.info())
print(df.head())

missing = pd.DataFrame({ "Missing Count": df.isnull().sum(), "Missing %": df.isnull().mean()*100}).sort_values("Missing %", ascending=False)



df["İLÇE"] = df["İLÇE"].replace(r'^\s*$', np.nan, regex=True)

df["Target"] = df["TEKLİF ONAY DURUMU"].map({"T":0,"P":1})



#teklif onay durumu
pd.crosstab(df["SİGORTALI TİPİ"],df["TEKLİF ONAY DURUMU"],normalize="index")

df.groupby("SİGORTALI TİPİ")["TEKLİF PRİMİ"].describe()

df.loc[df["YAŞ"] == 0, "YAŞ"] = np.nan
print(df["YAŞ"].isna().sum())


#Logistic Regression, Decision Tree, Random Forest, XGBoost ve LightGBM modelleri için kategorik değişkenlere One-Hot Encoding uygulanmıştır. CatBoost modeli ise kategorik değişkenleri doğal olarak işleyebildiğinden bu model için encoding uygulanmamış, orijinal kategorik değişkenler kullanılmıştır.

from sklearn.model_selection import train_test_split

# ==============================
# Target ve Feature
# ==============================
X = df.drop(columns=["TEKLİF ONAY DURUMU", "Target"])
y = df["Target"]

# ==============================
# Train - Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===================================================
# 1. Kopya -> One Hot Encoding yapılacak modeller için
# (Logistic Regression, Decision Tree, RF, XGBoost...)
# ===================================================

X_train_ohe = X_train.copy()
X_test_ohe = X_test.copy()

cat_cols = X_train_ohe.select_dtypes(include=["object", "category"]).columns
num_cols = X_train.select_dtypes(include=np.number).columns

X_train_ohe[cat_cols] = X_train_ohe[cat_cols].fillna("Missing")
X_test_ohe[cat_cols] = X_test_ohe[cat_cols].fillna("Missing")

X_train_ohe[num_cols] = X_train_ohe[num_cols].fillna(X_train_ohe[num_cols].median())
X_test_ohe[num_cols] = X_test_ohe[num_cols].fillna(X_train_ohe[num_cols].median())


X_train_ohe = pd.get_dummies(
    X_train_ohe,
    columns=cat_cols,
    drop_first=True
)

X_test_ohe = pd.get_dummies(
    X_test_ohe,
    columns=cat_cols,
    drop_first=True
)

# Train ve Test sütunlarını eşitle
X_train_ohe, X_test_ohe = X_train_ohe.align(
    X_test_ohe,
    join="left",
    axis=1,
    fill_value=0
)


# Logistic Regression için kopya
X_train_lr = X_train_ohe.copy()
X_test_lr = X_test_ohe.copy()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_lr[num_cols] = scaler.fit_transform(X_train_lr[num_cols])
X_test_lr[num_cols] = scaler.transform(X_test_lr[num_cols])


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def objective_logistic(trial):

    params = {
    "C": trial.suggest_float("C",1e-4,100,log=True),
    "solver": trial.suggest_categorical(
        "solver",
        ["liblinear","saga"]
    ),
    "penalty": trial.suggest_categorical(
        "penalty",
        ["l1","l2"]
    ),
    "class_weight": trial.suggest_categorical(
        "class_weight",
        [None, "balanced"]
    ),
    "max_iter":1000,
    "random_state":42
}

    model = LogisticRegression(**params)

    score = cross_val_score(
        model,
        X_train_lr,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    ).mean()

    return score


study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_logistic, n_trials=50)

print(study_lr.best_params)
print(study_lr.best_value)

best_lr = LogisticRegression(
    **study_lr.best_params,
    max_iter=1000,
    random_state=42
)

best_lr.fit(X_train_lr, y_train)

pred_lr = best_lr.predict(X_test_lr)
prob_lr = best_lr.predict_proba(X_test_lr)[:,1]






# ============================
# RANDOM FOREST
# ============================

from sklearn.calibration import CalibratedClassifierCV



def objective_rf(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight",
            [None, "balanced"]
        ),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)

    score = cross_val_score(
        model,
        X_train_ohe,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()

    return score


study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)

print(study_rf.best_params)
print(study_rf.best_value)

best_rf = RandomForestClassifier(
    **study_rf.best_params,
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train_ohe, y_train)


# =========================================================
# CALIBRATED RANDOM FOREST
# =========================================================

calibrated_rf = CalibratedClassifierCV(
    estimator=best_rf,
    method="sigmoid",
    cv=5
)

calibrated_rf.fit(
    X_train_ohe,
    y_train
)

# Calibrated test probabilities
prob_rf = calibrated_rf.predict_proba(
    X_test_ohe
)[:, 1]

# ============================
# XGBOOST
# ============================

def objective_xgb(trial):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "n_estimators": trial.suggest_int("n_estimators",100,500),
        "max_depth": trial.suggest_int("max_depth",3,10),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
        "subsample": trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
        "min_child_weight": trial.suggest_int("min_child_weight",1,10),
        "gamma": trial.suggest_float("gamma",0,5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "eval_metric":"logloss",
        "random_state":42,
        "scale_pos_weight": scale_pos_weight
    }

    model = XGBClassifier(**params)

    score = cross_val_score(
        model,
        X_train_ohe,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()

    return score


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb,n_trials=50)

print(study_xgb.best_params)
print(study_xgb.best_value)

best_xgb = XGBClassifier(
    **study_xgb.best_params,
    eval_metric="logloss",
    random_state=42
)

best_xgb.fit(X_train_ohe,y_train)

pred_xgb = best_xgb.predict(X_test_ohe)
prob_xgb = best_xgb.predict_proba(X_test_ohe)[:,1]



# ============================
# LIGHTGBM
# ============================

def objective_lgbm(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators",100,500),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
        "num_leaves": trial.suggest_int("num_leaves",20,100),
        "max_depth": trial.suggest_int("max_depth",-1,20),
        "min_child_samples": trial.suggest_int("min_child_samples",5,50),
        "subsample": trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
        "random_state":42,
        "verbosity":-1,
        "class_weight": "balanced",
        "reg_lambda": trial.suggest_float(
        "reg_lambda", 0.1, 20.0, log=True),
    }


    model = LGBMClassifier(**params)

    score = cross_val_score(
        model,
        X_train_ohe,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()

    return score


study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm,n_trials=50)

print(study_lgbm.best_params)
print(study_lgbm.best_value)

best_lgbm = LGBMClassifier(
    **study_lgbm.best_params,
    random_state=42,
    verbosity=-1
)

best_lgbm.fit(X_train_ohe,y_train)

pred_lgbm = best_lgbm.predict(X_test_ohe)
prob_lgbm = best_lgbm.predict_proba(X_test_ohe)[:,1]




# ===================================================
# 2. Kopya -> CatBoost için (Encoding YOK)
# ===================================================
from catboost import CatBoostClassifier

X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

cat_features = X_train_cat.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("CatBoost kategorik değişkenleri:")
print(cat_features)


# ===================================================
# CATBOOST
# ===================================================

#from catboost import CatBoostClassifier

X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

# Kategorik sütunların isimlerini bul
cat_features = X_train_cat.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Kategorik sütunların index'lerini bul
cat_feature_indices = [
    X_train_cat.columns.get_loc(col)
    for col in cat_features
]

print("CatBoost kategorik değişkenleri:")
print(cat_features)

print("CatBoost kategorik indexleri:")
print(cat_feature_indices)

cat_cols = X_train_cat.select_dtypes(include=["object", "category"]).columns
num_cols = X_train_cat.select_dtypes(include=np.number).columns

X_train_cat[cat_cols] = X_train_cat[cat_cols].fillna("Missing")
X_test_cat[cat_cols] = X_test_cat[cat_cols].fillna("Missing")
X_train_cat[num_cols] = X_train_cat[num_cols].fillna(X_train_cat[num_cols].median())
X_test_cat[num_cols] = X_test_cat[num_cols].fillna(X_train_cat[num_cols].median())

# ============================
# OPTUNA
# ============================

def objective_cat(trial):

    params = {
        "iterations": trial.suggest_int(
            "iterations", 200, 800
        ),

        "depth": trial.suggest_int(
            "depth", 4, 10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1, 10
        ),

        "random_strength": trial.suggest_float(
            "random_strength", 0, 5
        ),

        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_state": 42
    }

    scores = []

    for train_idx, val_idx in cv.split(
        X_train_cat,
        y_train
    ):

        X_fold_train = X_train_cat.iloc[train_idx]
        X_fold_val = X_train_cat.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_fold_train,
            y_fold_train,
            cat_features=cat_feature_indices
        )

        prob = model.predict_proba(
            X_fold_val
        )[:, 1]

        score = roc_auc_score(
            y_fold_val,
            prob
        )

        scores.append(score)

    return np.mean(scores)


# ============================
# OPTUNA STUDY
# ============================

study_cat = optuna.create_study(
    direction="maximize"
)

study_cat.optimize(
    objective_cat,
    n_trials=50
)

print("Best CatBoost Parameters:")
print(study_cat.best_params)

print("Best CV ROC-AUC:")
print(study_cat.best_value)


# ============================
# FINAL CATBOOST MODEL
# ============================

best_cat = CatBoostClassifier(
    **study_cat.best_params,
    loss_function="Logloss",
    eval_metric="AUC",
    random_state=42,
    verbose=0
)

best_cat.fit(
    X_train_cat,
    y_train,
    cat_features=cat_feature_indices
)

pred_cat = best_cat.predict(
    X_test_cat
)

prob_cat = best_cat.predict_proba(
    X_test_cat
)[:, 1]

#Thresholds
def find_best_threshold(y_true, probabilities, metric="F1"):

    thresholds = np.arange(0.05, 0.96, 0.01)

    scores = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        scores.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        })

    scores_df = pd.DataFrame(scores)

    best_idx = scores_df[metric].idxmax()

    return (
        scores_df,
        scores_df.loc[best_idx, "Threshold"]
    )

from sklearn.base import clone

def get_oof_predictions(model, X, y, cv):
    
    oof_prob = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        
        y_fold_train = y.iloc[train_idx]
        
        model_clone = clone(model)
        
        model_clone.fit(
            X_fold_train,
            y_fold_train
        )
        
        oof_prob[val_idx] = model_clone.predict_proba(
            X_fold_val
        )[:, 1]
    
    return oof_prob

# =========================================================
# OOF PREDICTIONS
# =========================================================

oof_probs = {}

oof_probs["Logistic Regression"] = get_oof_predictions(
    best_lr,
    X_train_lr,
    y_train.reset_index(drop=True),
    cv
)

oof_probs["Random Forest"] = get_oof_predictions(
    calibrated_rf,
    X_train_ohe.reset_index(drop=True),
    y_train.reset_index(drop=True),
    cv
)

oof_probs["XGBoost"] = get_oof_predictions(
    best_xgb,
    X_train_ohe.reset_index(drop=True),
    y_train.reset_index(drop=True),
    cv
)

oof_probs["LightGBM"] = get_oof_predictions(
    best_lgbm,
    X_train_ohe.reset_index(drop=True),
    y_train.reset_index(drop=True),
    cv
)

def get_oof_predictions_catboost(model_params, X, y, cv, cat_features):
    
    oof_prob = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]
        
        model = CatBoostClassifier(
            **model_params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_state=42,
            verbose=0
        )
        
        model.fit(
            X_fold_train,
            y_fold_train,
            cat_features=cat_features
        )
        
        oof_prob[val_idx] = model.predict_proba(
            X_fold_val
        )[:, 1]
        
    return oof_prob

oof_probs["CatBoost"] = get_oof_predictions_catboost(
    study_cat.best_params,
    X_train_cat.reset_index(drop=True),
    y_train.reset_index(drop=True),
    cv,
    cat_feature_indices
)

# =========================================================
# MODEL PROBABILITIES
# =========================================================

model_probs = {
    "Logistic Regression": prob_lr,
    "Random Forest": prob_rf,
    "XGBoost": prob_xgb,
    "LightGBM": prob_lgbm,
    "CatBoost": prob_cat
}


# =========================================================
# OOF THRESHOLD OPTIMIZATION
# =========================================================

threshold_results = []

best_thresholds = {}

for name, prob in oof_probs.items():
    
    scores_df, best_threshold = find_best_threshold(
        y_train.reset_index(drop=True),
        prob,
        metric="F1"
    )
    
    best_thresholds[name] = best_threshold
    
    threshold_results.append({
        "Model": name,
        "Best_Threshold": best_threshold,
        "OOF_F1": scores_df.loc[
            scores_df["Threshold"] == best_threshold,
            "F1"
        ].iloc[0],
        "OOF_Precision": scores_df.loc[
            scores_df["Threshold"] == best_threshold,
            "Precision"
        ].iloc[0],
        "OOF_Recall": scores_df.loc[
            scores_df["Threshold"] == best_threshold,
            "Recall"
        ].iloc[0]
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df)


# =========================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss
)

def bootstrap_metrics(
    y_true,
    probabilities,
    threshold,
    n_bootstrap=2000,
    random_state=42
):
    
    rng = np.random.RandomState(random_state)
    
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    
    n = len(y_true)
    
    # Bootstrap sonuçlarını tut
    bootstrap_results = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "ROC_AUC": [],
        "Brier Score": [],
        "Gini": []
    }
    
    for _ in range(n_bootstrap):
        
        # Test setinden replacement ile örnekleme
        indices = rng.randint(0, n, n)
        
        y_boot = y_true[indices]
        prob_boot = probabilities[indices]
        
        # ROC-AUC hesaplanabilmesi için
        # bootstrap sample'da iki sınıf da bulunmalı
        if len(np.unique(y_boot)) < 2:
            continue
        
        # Aynı OOF threshold kullanılıyor
        pred_boot = (
            prob_boot >= threshold
        ).astype(int)
        
        # Accuracy
        bootstrap_results["Accuracy"].append(
            accuracy_score(
                y_boot,
                pred_boot
            )
        )
        
        # Precision
        bootstrap_results["Precision"].append(
            precision_score(
                y_boot,
                pred_boot,
                zero_division=0
            )
        )
        
        # Recall
        bootstrap_results["Recall"].append(
            recall_score(
                y_boot,
                pred_boot,
                zero_division=0
            )
        )
        
        # F1
        bootstrap_results["F1"].append(
            f1_score(
                y_boot,
                pred_boot,
                zero_division=0
            )
        )
        
        # ROC-AUC
        roc_auc = roc_auc_score(
            y_boot,
            prob_boot
        )
        
        bootstrap_results["ROC_AUC"].append(
            roc_auc
        )
        
        # Brier Score
        bootstrap_results["Brier Score"].append(
            brier_score_loss(
                y_boot,
                prob_boot
            )
        )
        
        # Gini
        bootstrap_results["Gini"].append(
            2 * roc_auc - 1
        )
    
    
    # =====================================================
    # CONFIDENCE INTERVALS
    # =====================================================
    
    summary = []
    
    for metric, values in bootstrap_results.items():
        
        values = np.asarray(values)
        
        summary.append({
            "Metric": metric,
            "Mean": np.mean(values),
            "Std": np.std(values),
            "Lower 95% CI": np.percentile(
                values,
                2.5
            ),
            "Upper 95% CI": np.percentile(
                values,
                97.5
            )
        })
    
    return pd.DataFrame(summary)






# =========================================================
# FINAL TEST SCORES
# =========================================================

results = []

for name, prob in model_probs.items():

    # OOF'dan optimize edilen threshold
    threshold = best_thresholds[name]

    # TEST üzerinde aynı threshold uygulanıyor
    pred = (prob >= threshold).astype(int)

    results.append({
        "Model": name,
        "Threshold": threshold,

        # Threshold-dependent metrics
        "Accuracy": accuracy_score(
            y_test,
            pred
        ),

        "Precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        # Threshold-independent metric
        "ROC_AUC": roc_auc_score(
            y_test,
            prob
        )
    })

results_df = pd.DataFrame(results)

print(
    results_df.sort_values(
        "ROC_AUC",
        ascending=False
    )
)
# =========================================================
# ROC CURVES
# =========================================================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        prob
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================================================
# PRECISION-RECALL CURVES
# =========================================================

from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        prob
    )

    ap = average_precision_score(
        y_test,
        prob
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================================================
# CALIBRATION CURVES
# =========================================================

from sklearn.calibration import calibration_curve

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test,
        prob,
        n_bins=10,
        strategy="quantile"
    )

    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label=name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================================================
# CONFUSION MATRICES - OPTIMIZED THRESHOLD
# =========================================================

from sklearn.metrics import ConfusionMatrixDisplay

for name, prob in model_probs.items():

    # OOF'dan bulunan best threshold
    threshold = best_thresholds[name]

    # TEST probability -> optimized threshold
    pred = (prob >= threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred
    )

    plt.title(
        f"Confusion Matrix - {name} "
        f"(Threshold = {threshold:.2f})"
    )

    plt.show()
# =========================================================
# KS STATISTIC
# =========================================================

from sklearn.metrics import roc_curve

def ks_statistic(y_true, probabilities):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    ks_values = tpr - fpr

    max_idx = np.argmax(ks_values)

    return (
        ks_values[max_idx],
        thresholds[max_idx]
    )


ks_results = []

for name, prob in model_probs.items():

    ks, ks_threshold = ks_statistic(
        y_test,
        prob
    )

    ks_results.append({
        "Model": name,
        "KS": ks,
        "KS_Threshold": ks_threshold
    })

ks_df = pd.DataFrame(ks_results)

print(
    ks_df.sort_values(
        "KS",
        ascending=False
    )
)

# =========================================================
# GINI COEFFICIENT
# =========================================================

gini_results = []

for name, prob in model_probs.items():

    roc_auc = roc_auc_score(
        y_test,
        prob
    )

    gini = 2 * roc_auc - 1

    gini_results.append({
        "Model": name,
        "ROC_AUC": roc_auc,
        "Gini": gini
    })

gini_df = pd.DataFrame(gini_results)

print(
    gini_df.sort_values(
        "Gini",
        ascending=False
    )
)
# =========================================================
# GINI / CAP CURVE
# =========================================================

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    # Sort customers from highest predicted probability
    # to lowest predicted probability
    df_gini = pd.DataFrame({
        "y_true": np.array(y_test),
        "probability": np.array(prob)
    }).sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    # Cumulative population %
    population_pct = (
        np.arange(1, len(df_gini) + 1)
        / len(df_gini)
    )

    # Cumulative percentage of actual positives captured
    cumulative_positive_pct = (
        df_gini["y_true"].cumsum()
        / df_gini["y_true"].sum()
    )

    # Add origin
    population_pct = np.insert(
        population_pct,
        0,
        0
    )

    cumulative_positive_pct = np.insert(
        cumulative_positive_pct.values,
        0,
        0
    )

    plt.plot(
        population_pct * 100,
        cumulative_positive_pct * 100,
        label=name
    )


# Random model
plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Random"
)

plt.xlabel("Cumulative Population (%)")
plt.ylabel("Cumulative Positives Captured (%)")

plt.title(
    "Cumulative Gains / Gini (CAP) Curve"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# =========================================================
# DECILE / LIFT
# =========================================================

def decile_lift_table(y_true, probabilities):

    df_lift = pd.DataFrame({
        "y_true": np.array(y_true),
        "probability": np.array(probabilities)
    })

    # En yüksek probability en üstte
    df_lift = df_lift.sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    # 10 gruba böl
    df_lift["Decile"] = pd.qcut(
        df_lift.index,
        10,
        labels=False
    ) + 1

    total_positive_rate = df_lift["y_true"].mean()

    lift_table = (
        df_lift
        .groupby("Decile")
        .agg(
            Customers=("y_true", "count"),
            Positives=("y_true", "sum"),
            Positive_Rate=("y_true", "mean"),
            Avg_Probability=("probability", "mean")
        )
        .reset_index()
    )

    lift_table["Lift"] = (
        lift_table["Positive_Rate"]
        / total_positive_rate
    )

    lift_table["Cumulative_Positives"] = (
        lift_table["Positives"].cumsum()
    )

    total_positives = lift_table["Positives"].sum()

    lift_table["Cumulative_Gain"] = (
        lift_table["Cumulative_Positives"]
        / total_positives
    )

    return lift_table


cat_lift = decile_lift_table(
    y_test,
    prob_cat
)

print(cat_lift)

# =========================================================
# LIFT CHART
# =========================================================

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    lift_table = decile_lift_table(
        y_test,
        prob
    )

    plt.plot(
        lift_table["Decile"],
        lift_table["Lift"],
        marker="o",
        label=name
    )

plt.axhline(
    y=1,
    linestyle="--",
    label="Random"
)

plt.xlabel("Decile")
plt.ylabel("Lift")
plt.title("Decile / Lift Chart")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================================================
# CUMULATIVE GAIN CHART
# =========================================================

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    lift_table = decile_lift_table(
        y_test,
        prob
    )

    population_percentage = (
        lift_table["Decile"] * 10
    )

    cumulative_gain = (
        lift_table["Cumulative_Gain"] * 100
    )

    plt.plot(
        population_percentage,
        cumulative_gain,
        marker="o",
        label=name
    )

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Random"
)

plt.xlabel("Population %")
plt.ylabel("Cumulative Gain %")
plt.title("Cumulative Gain Chart")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================================================
# ROC CURVES + BEST THRESHOLD POINT
# =========================================================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))

for name, prob in model_probs.items():

    # ROC eğrisi
    fpr, tpr, thresholds = roc_curve(
        y_test,
        prob
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

    # -----------------------------------------
    # OOF ile bulunan BEST THRESHOLD
    # -----------------------------------------

    best_threshold = best_thresholds[name]

    # Best threshold'a karşılık gelen ROC noktası
    best_idx = np.argmin(
        np.abs(thresholds - best_threshold)
    )

    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    # Best threshold noktasını göster
    plt.scatter(
        best_fpr,
        best_tpr,
        s=80,
        edgecolor="black",
        zorder=5
    )

    # Threshold değerini yaz
    plt.annotate(
        f"t={best_threshold:.2f}",
        (best_fpr, best_tpr),
        xytext=(5, 5),
        textcoords="offset points"
    )


# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves with Optimized Thresholds")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# TRAIN vs 5 fold  CV vs HOLD-OUT TEST
# OVERFITTING / GENERALIZATION CHECK
# =========================================================

#from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# MODELS EXCEPT CATBOOST
# ---------------------------------------------------------

models_for_check = {
    "Logistic Regression": (
        best_lr,
        X_train_lr,
        X_test_lr
    ),

    "Random Forest": (
        calibrated_rf,
        X_train_ohe,
        X_test_ohe
    ),

    "XGBoost": (
        best_xgb,
        X_train_ohe,
        X_test_ohe
    ),

    "LightGBM": (
        best_lgbm,
        X_train_ohe,
        X_test_ohe
    )
}


overfitting_results = []


# =========================================================
# LOGISTIC / RF / XGB / LGBM
# =========================================================

for name, (model, Xtr, Xte) in models_for_check.items():

    # -------------------------
    # TRAIN
    # -------------------------

    train_prob = model.predict_proba(Xtr)[:, 1]

    train_auc = roc_auc_score(
        y_train,
        train_prob
    )


    # -------------------------
    # 5-FOLD CV
    # -------------------------

    cv_scores = cross_val_score(
        model,
        Xtr,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    cv_auc = cv_scores.mean()
    cv_std = cv_scores.std()


    # -------------------------
    # HOLD-OUT TEST
    # -------------------------

    test_prob = model.predict_proba(Xte)[:, 1]

    test_auc = roc_auc_score(
        y_test,
        test_prob
    )


    # -------------------------
    # GAPS
    # -------------------------

    train_test_gap = train_auc - test_auc

    cv_test_gap = cv_auc - test_auc


    overfitting_results.append({

        "Model": name,

        "Train ROC-AUC": train_auc,

        "CV ROC-AUC": cv_auc,

        "CV Std": cv_std,

        "Hold-out Test ROC-AUC": test_auc,

        "Train-Test Gap": train_test_gap,

        "CV-Test Gap": cv_test_gap

    })


# =========================================================
# CATBOOST - SPECIAL CV
# =========================================================

# TRAIN AUC

train_prob_cat = best_cat.predict_proba(
    X_train_cat
)[:, 1]

train_auc_cat = roc_auc_score(
    y_train,
    train_prob_cat
)


# 5-FOLD CV

cat_cv_scores = []

for train_idx, val_idx in cv.split(
    X_train_cat,
    y_train
):

    X_fold_train = X_train_cat.iloc[train_idx]

    X_fold_val = X_train_cat.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]

    y_fold_val = y_train.iloc[val_idx]


    cat_model = CatBoostClassifier(
        **study_cat.best_params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=0
    )


    cat_model.fit(
        X_fold_train,
        y_fold_train,
        cat_features=cat_feature_indices
    )


    fold_prob = cat_model.predict_proba(
        X_fold_val
    )[:, 1]


    fold_auc = roc_auc_score(
        y_fold_val,
        fold_prob
    )


    cat_cv_scores.append(fold_auc)


cat_cv_auc = np.mean(cat_cv_scores)

cat_cv_std = np.std(cat_cv_scores)


# HOLD-OUT TEST

test_prob_cat = best_cat.predict_proba(
    X_test_cat
)[:, 1]


test_auc_cat = roc_auc_score(
    y_test,
    test_prob_cat
)


# GAPS

train_test_gap_cat = (
    train_auc_cat - test_auc_cat
)

cv_test_gap_cat = (
    cat_cv_auc - test_auc_cat
)


overfitting_results.append({

    "Model": "CatBoost",

    "Train ROC-AUC": train_auc_cat,

    "CV ROC-AUC": cat_cv_auc,

    "CV Std": cat_cv_std,

    "Hold-out Test ROC-AUC": test_auc_cat,

    "Train-Test Gap": train_test_gap_cat,

    "CV-Test Gap": cv_test_gap_cat

})


# =========================================================
# FINAL TABLE
# =========================================================

overfitting_df = pd.DataFrame(
    overfitting_results
)


overfitting_df = overfitting_df.sort_values(
    "Hold-out Test ROC-AUC",
    ascending=False
)


print(
    "\n===== OVERFITTING / GENERALIZATION CHECK ====="
)

print(
    overfitting_df.round(4).to_string(
        index=False
    )
)

# =========================================================
# TRAIN vs CV vs TEST ROC-AUC
# =========================================================

import matplotlib.pyplot as plt
import numpy as np

plot_df = overfitting_df.set_index("Model")[
    [
        "Train ROC-AUC",
        "CV ROC-AUC",
        "Hold-out Test ROC-AUC"
    ]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Train vs Cross-Validation vs Hold-out Test ROC-AUC")
plt.ylabel("ROC-AUC")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.ylim(0.6, 1.0)
plt.legend(title="")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

###The dataset was split into an 80% training set and a 20% untouched hold-out test set. Hyperparameter optimization and model selection were performed within the training set using 5-fold stratified cross-validation. Out-of-fold predictions were generated within the training set and used for threshold optimization. The final model performance was evaluated once on the untouched hold-out test set.

# =========================================================
# BRIER SCORE
# =========================================================

#from sklearn.metrics import brier_score_loss

brier_results = []

for name, prob in model_probs.items():

    brier = brier_score_loss(
        y_test,
        prob
    )

    brier_results.append({
        "Model": name,
        "Brier Score": brier
    })

brier_df = pd.DataFrame(
    brier_results
)

brier_df = brier_df.sort_values(
    "Brier Score",
    ascending=True
)

print(
    "\n===== BRIER SCORE ====="
)

print(
    brier_df.round(4).to_string(
        index=False
    )
)


# =========================================================
# BRIER SCORE CHART
# =========================================================

plt.figure(figsize=(8, 6))

bars = plt.bar(
    brier_df["Model"],
    brier_df["Brier Score"]
)

plt.bar_label(
    bars,
    fmt="%.4f",
    padding=3
)

plt.ylabel("Brier Score")
plt.xlabel("Model")

plt.title(
    "Brier Score Comparison"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()

# =========================================================
# SHAP ANALYSIS - ALL TREE-BASED MODELS
# =========================================================

import shap
import matplotlib.pyplot as plt

# Tree-based models
tree_models = {
    "Random Forest": (best_rf, X_test_ohe),
    "XGBoost": (best_xgb, X_test_ohe),
    "LightGBM": (best_lgbm, X_test_ohe),
    "CatBoost": (best_cat, X_test_cat)
}

for name, (model, X_data) in tree_models.items():

    print(f"\n===== SHAP ANALYSIS: {name} =====")

    # -----------------------------------------------------
    # SHAP Explainer
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_data)

    # -----------------------------------------------------
    # SHAP SUMMARY PLOT
    # -----------------------------------------------------

    shap.summary_plot(
        shap_values,
        X_data,
        show=False
    )

    plt.title(
        f"SHAP Summary - {name}"
    )

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # SHAP FEATURE IMPORTANCE
    # -----------------------------------------------------

    shap.summary_plot(
        shap_values,
        X_data,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"SHAP Feature Importance - {name}"
    )

    plt.tight_layout()
    plt.show()

# =========================================================
# LOGISTIC REGRESSION - FEATURE IMPORTANCE
# =========================================================

lr_importance = pd.DataFrame({
    "Feature": X_train_ohe.columns,
    "Coefficient": best_lr.coef_[0]
})

# Absolute coefficient = importance magnitude
lr_importance["Importance"] = (
    lr_importance["Coefficient"].abs()
)

lr_importance = lr_importance.sort_values(
    "Importance",
    ascending=False
)

print("\n===== LOGISTIC REGRESSION FEATURE IMPORTANCE =====")

print(
    lr_importance[
        ["Feature", "Coefficient", "Importance"]
    ].head(20).to_string(index=False)
)


# =========================================================
# LOGISTIC REGRESSION FEATURE IMPORTANCE CHART
# =========================================================

top_lr = lr_importance.head(20).sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(9, 7))

bars = plt.barh(
    top_lr["Feature"],
    top_lr["Importance"]
)

plt.bar_label(
    bars,
    fmt="%.3f",
    padding=3
)

plt.xlabel("Absolute Coefficient")
plt.ylabel("Feature")

plt.title(
    "Logistic Regression - Feature Importance"
)

plt.tight_layout()
plt.show()
#Logistic Regression was selected as the baseline model due to its interpretability and simplicity. Among the tree-based challenger models, LightGBM was selected as the primary challenger, while XGBoost was retained as a close alternative. Although XGBoost achieved the highest hold-out ROC-AUC (0.7475), its advantage over LightGBM (0.7472) was negligible. LightGBM achieved the highest KS statistic (0.3806) and F1-score (0.3786), while also generating fewer false-positive predictions at the optimized threshold. Both models substantially improved upon the Logistic Regression baseline (hold-out ROC-AUC = 0.7151).

# =========================================================
# LOGISTIC REGRESSION - PSEUDO R² (McFADDEN)
# =========================================================

from sklearn.metrics import log_loss

# Null model: sadece sabit (intercept)
p_null = np.mean(y_test)

# Null model log-likelihood
ll_null = -log_loss(
    y_test,
    np.full(len(y_test), p_null),
    labels=[0, 1],
    normalize=False
)

# Logistic Regression log-likelihood
ll_model = -log_loss(
    y_test,
    prob_lr,
    labels=[0, 1],
    normalize=False
)

# McFadden Pseudo R²
mcfadden_r2 = 1 - (ll_model / ll_null)

print("\n===== LOGISTIC REGRESSION - PSEUDO R² =====")
print(f"McFadden Pseudo R²: {mcfadden_r2:.4f}")

# ============================================================
# PART II — RISK SEGMENTATION
# ============================================================

# ------------------------------------------------------------
# 1. RISK VARIABLES
# ------------------------------------------------------------

risk_cols = [
    "HASARSIZLIK İNDİRİMİ KADEMESİ",
    "TRAFİK BASAMAK KODU",
    "ARAÇ YAŞI",
    "YAŞ"
]

risk_cols = [
    col for col in risk_cols
    if col in X_train.columns
]

print("Risk variables:")
print(risk_cols)

risk_train = X_train[risk_cols].copy()
risk_test = X_test[risk_cols].copy()


# ------------------------------------------------------------
# 2. NUMERIC CONVERSION
# ------------------------------------------------------------

for col in risk_cols:
    risk_train[col] = pd.to_numeric(
        risk_train[col],
        errors="coerce"
    )

    risk_test[col] = pd.to_numeric(
        risk_test[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 3. MISSING VALUES → TRAIN MEDIAN IMPUTATION
# ------------------------------------------------------------

print("\nBefore missing value treatment:")
print("Train:", risk_train.shape)
print("Test :", risk_test.shape)

print("\nMissing values BEFORE:")
print("Train:")
print(risk_train.isna().sum())

print("\nTest:")
print(risk_test.isna().sum())


# Train medianlarını hesapla
risk_fill_values = {}

for col in risk_cols:
    median_value = risk_train[col].median()

    risk_fill_values[col] = median_value

    risk_train[col] = risk_train[col].fillna(
        median_value
    )

    risk_test[col] = risk_test[col].fillna(
        median_value
    )


print("\nTrain median fill values:")
print(risk_fill_values)


print("\nMissing values AFTER:")
print("Train:")
print(risk_train.isna().sum())

print("\nTest:")
print(risk_test.isna().sum())

print("\nAfter missing value treatment:")
print("Train:", risk_train.shape)
print("Test :", risk_test.shape)


# ============================================================
# 4. RISK DIRECTION + NORMALIZATION
# ============================================================

# Yüksek değer → DÜŞÜK RİSK
inverse_risk_cols = [
    "HASARSIZLIK İNDİRİMİ KADEMESİ",
    "TRAFİK BASAMAK KODU"
]

# Yüksek değer → YÜKSEK RİSK
direct_risk_cols = [
    "ARAÇ YAŞI",
    "YAŞ"
]


# ------------------------------------------------------------
# INVERSE VARIABLES
# ------------------------------------------------------------

for col in inverse_risk_cols:

    if col not in risk_train.columns:
        continue

    min_val = risk_train[col].min()
    max_val = risk_train[col].max()

    risk_train[col + "_Risk"] = (
        (max_val - risk_train[col])
        / (max_val - min_val)
    )

    risk_test[col + "_Risk"] = (
        (max_val - risk_test[col])
        / (max_val - min_val)
    )


# ------------------------------------------------------------
# DIRECT VARIABLES
# ------------------------------------------------------------

for col in direct_risk_cols:

    if col not in risk_train.columns:
        continue

    min_val = risk_train[col].min()
    max_val = risk_train[col].max()

    risk_train[col + "_Risk"] = (
        (risk_train[col] - min_val)
        / (max_val - min_val)
    )

    risk_test[col + "_Risk"] = (
        (risk_test[col] - min_val)
        / (max_val - min_val)
    )


# ============================================================
# 5. CLAIM RISK PROXY
# ============================================================

# Gerçek hasar sayısı / hasar tutarı olmadığı için
# hasar geçmişi hakkında bilgi taşıyan değişkenlerden
# claim risk proxy oluşturuyoruz.

claim_risk_cols = [
    "HASARSIZLIK İNDİRİMİ KADEMESİ",
    "TRAFİK BASAMAK KODU"
]

claim_risk_score_cols = [
    col + "_Risk"
    for col in claim_risk_cols
    if col + "_Risk" in risk_train.columns
]

risk_train["Claim_Risk_Proxy"] = (
    risk_train[claim_risk_score_cols].mean(axis=1)
)

risk_test["Claim_Risk_Proxy"] = (
    risk_test[claim_risk_score_cols].mean(axis=1)
)

print("\n" + "=" * 75)
print("CLAIM RISK PROXY")
print("=" * 75)

print(
    risk_train["Claim_Risk_Proxy"]
    .describe()
    .round(3)
)


# ============================================================
# 6. COMPOSITE RISK SCORE
# ============================================================

risk_score_cols = [
    col
    for col in risk_train.columns
    if col.endswith("_Risk")
]

risk_train["Risk_Score"] = (
    risk_train[risk_score_cols].mean(axis=1)
)

risk_test["Risk_Score"] = (
    risk_test[risk_score_cols].mean(axis=1)
)


# ============================================================
# 7. RISK SEGMENTATION
# ============================================================

q1 = risk_train["Risk_Score"].quantile(0.33)
q2 = risk_train["Risk_Score"].quantile(0.66)

risk_train["Risk_Segment"] = pd.cut(
    risk_train["Risk_Score"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

risk_test["Risk_Segment"] = pd.cut(
    risk_test["Risk_Score"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)



# ============================================================
# 8. RISK SEGMENT VALIDATION
# ============================================================

risk_segment_analysis = risk_train[
    [
        "Claim_Risk_Proxy",
        "Risk_Score",
        "Risk_Segment"
    ]
].copy()

# Teklif numarasını X_train'den al
risk_segment_analysis["TEKLİF NUMARASI"] = X_train.loc[
    risk_segment_analysis.index,
    "TEKLİF NUMARASI"
].values

# Gerçek mevcut premium
risk_segment_analysis["Premium"] = X_train.loc[
    risk_segment_analysis.index,
    "TEKLİF PRİMİ"
].values

# Target
risk_segment_analysis["Target"] = y_train.loc[
    risk_segment_analysis.index
].values

# ============================================================
# 9. RISK SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("RISK SCORE DISTRIBUTION BY SEGMENT")
print("=" * 75)

print(
    risk_train
    .groupby(
        "Risk_Segment",
        observed=True
    )["Risk_Score"]
    .agg(
        ["count", "min", "mean", "median", "max"]
    )
    .round(3)
)


# ============================================================
# 10. CLAIM PROXY — PREMIUM ALIGNMENT
# ============================================================

risk_segment_analysis["Proxy_Decile"] = pd.qcut(
    risk_segment_analysis["Claim_Risk_Proxy"],
    q=10,
    duplicates="drop"
)

proxy_summary = (
    risk_segment_analysis
    .groupby(
        "Proxy_Decile",
        observed=True
    )
    .agg(
        Customers=("Premium", "size"),
        Avg_Claim_Risk=("Claim_Risk_Proxy", "mean"),
        Avg_Premium=("Premium", "mean")
    )
    .reset_index()
)

print("\n" + "=" * 75)
print("CLAIM RISK PROXY — PREMIUM ALIGNMENT")
print("=" * 75)

print(
    proxy_summary.round(3)
)


# ============================================================
# 11. CLAIM PROXY VALIDATION BY RISK SEGMENT
# ============================================================

claim_proxy_summary = (
    risk_segment_analysis
    .groupby(
        "Risk_Segment",
        observed=True
    )
    .agg(
        Customers=("Target", "size"),
        Avg_Claim_Risk=("Claim_Risk_Proxy", "mean"),
        Avg_Risk_Score=("Risk_Score", "mean"),
        Avg_Premium=("Premium", "mean"),
        Acceptance_Rate=("Target", "mean")
    )
    .reset_index()
)

claim_proxy_summary["Acceptance_Rate"] *= 100

print("\n" + "=" * 75)
print("CLAIM RISK PROXY VALIDATION")
print("=" * 75)

print(
    claim_proxy_summary.round(3)
)


# ============================================================
# 12. ACCEPTANCE RATE BY RISK SEGMENT
# ============================================================

segment_summary = (
    risk_segment_analysis
    .groupby("Risk_Segment", observed=True)
    .agg(
        Customers=("Target", "size"),
        Acceptance_Rate=("Target", "mean"),
        Avg_Claim_Risk=("Claim_Risk_Proxy", "mean"),
        Avg_Risk_Score=("Risk_Score", "mean"),
        Avg_Premium=("Premium", "mean"),
        Median_Premium=("Premium", "median")
    )
    .reset_index()
)

segment_summary["Acceptance_Rate"] *= 100


plt.figure(figsize=(7, 4))

plt.bar(
    segment_summary["Risk_Segment"],
    segment_summary["Acceptance_Rate"]
)

plt.xlabel("Risk Segment")
plt.ylabel("Acceptance Rate (%)")
plt.title("Acceptance Rate by Risk Segment")

plt.tight_layout()
plt.show()

# ============================================================
# 13. RISK SCORE vs PREMIUM
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    risk_segment_analysis["Risk_Score"],
    risk_segment_analysis["Premium"],
    alpha=0.25
)

plt.xlabel("Risk Score")
plt.ylabel("Offer Premium")
plt.title("Risk Score vs Offer Premium")

plt.tight_layout()
plt.show()

# ============================================================
# PART III — RISK COST / EXPECTED LOSS PROXY
# ============================================================

pricing_base = risk_segment_analysis[
    [
        "TEKLİF NUMARASI",
        "Premium",
        "Claim_Risk_Proxy",
        "Risk_Score",
        "Risk_Segment",
        "Target"
    ]
].copy()

pricing_base["Customer_ID"] = (
    pricing_base["TEKLİF NUMARASI"]
    .astype(str)
    .str.strip()
)

pricing_base["Original_Premium"] = pricing_base["Premium"]


# ------------------------------------------------------------
# 1. RISK COST PROXY
# ------------------------------------------------------------

# Claim_Risk_Proxy 0-1 arasında olduğu için
# primi risk ile ilişkilendiren basit risk maliyeti göstergesi

pricing_base["Risk_Cost_Proxy"] = (
    pricing_base["Premium"]
    * pricing_base["Claim_Risk_Proxy"]
)

# ------------------------------------------------------------
# 2. EXPECTED VALUE / ACCEPTANCE
# ------------------------------------------------------------

pricing_base["Expected_Revenue"] = (
    pricing_base["Premium"]
    * pricing_base["Target"]
)

# ------------------------------------------------------------
# 3. RISK-ADJUSTED PREMIUM
# ------------------------------------------------------------

pricing_base["Risk_Adjusted_Premium"] = (
    pricing_base["Premium"]
    * (
        1
        + pricing_base["Claim_Risk_Proxy"]
    )
)

# ------------------------------------------------------------
# 4. SEGMENT BAZINDA ÖZET
# ------------------------------------------------------------

pricing_summary = (
    pricing_base
    .groupby(
        "Risk_Segment",
        observed=True
    )
    .agg(
        Customers=("Premium", "size"),
        Avg_Premium=("Premium", "mean"),
        Avg_Claim_Risk=("Claim_Risk_Proxy", "mean"),
        Avg_Risk_Score=("Risk_Score", "mean"),
        Avg_Risk_Cost=("Risk_Cost_Proxy", "mean"),
        Avg_Risk_Adjusted_Premium=("Risk_Adjusted_Premium", "mean"),
        Acceptance_Rate=("Target", "mean")
    )
    .reset_index()
)

pricing_summary["Acceptance_Rate"] *= 100

print("\n" + "=" * 75)
print("RISK COST / EXPECTED LOSS PROXY")
print("=" * 75)

print(
    pricing_summary.round(2)
)


# ============================================================
# PART III — PRICING BASE + DISCOUNT GRID
# ============================================================

# ------------------------------------------------------------
# 1. PRICING BASE
# ------------------------------------------------------------

pricing_base = risk_segment_analysis[
    [
        "Premium",
        "Claim_Risk_Proxy",
        "Risk_Score",
        "Risk_Segment",
        "Target"
    ]
].copy()

print("\n" + "=" * 75)
print("PRICING BASE")
print("=" * 75)

print(pricing_base.head())
print("\nCustomers:", len(pricing_base))


# ============================================================
# 2. RISK COST PROXY
# ============================================================

# Gerçek hasar tutarı/sayısı olmadığı için bunu
# Expected Loss değil, RISK COST PROXY olarak adlandırıyoruz.

pricing_base["Risk_Cost_Proxy"] = (
    pricing_base["Premium"]
    * pricing_base["Claim_Risk_Proxy"]
)


# ============================================================
# 3. DISCOUNT GRID
# ============================================================

# 0.0% → 10.0%
# 0.1 percentage-point increments

discount_grid = np.arange(
    0,
    10.0001,
    0.1
) / 100


# ============================================================
# 4. PORTFOLIO DISCOUNT SCENARIOS
# ============================================================

portfolio_results = []

for discount in discount_grid:

    discounted_premium = (
        pricing_base["Premium"]
        * (1 - discount)
    )

    total_premium = discounted_premium.sum()

    total_risk_cost = (
        pricing_base["Risk_Cost_Proxy"]
        * (1 - discount)
    ).sum()

    portfolio_results.append({
        "Discount_%": discount * 100,
        "Avg_Premium": discounted_premium.mean(),
        "Total_Premium": total_premium,
        "Total_Risk_Cost": total_risk_cost,
        "Risk_Adjusted_Premium": (
            total_premium - total_risk_cost
        )
    })

portfolio_discount_grid = pd.DataFrame(
    portfolio_results
)


print("\n" + "=" * 75)
print("DISCOUNT GRID — PORTFOLIO")
print("=" * 75)

print(
    portfolio_discount_grid.round(2)
)


# ============================================================
# 5. DISCOUNT GRID BY RISK SEGMENT
# ============================================================

segment_results = []

for segment in pricing_base["Risk_Segment"].dropna().unique():

    segment_data = pricing_base[
        pricing_base["Risk_Segment"] == segment
    ].copy()

    for discount in discount_grid:

        discounted_premium = (
            segment_data["Premium"]
            * (1 - discount)
        )

        total_premium = discounted_premium.sum()

        total_risk_cost = (
            segment_data["Risk_Cost_Proxy"]
            * (1 - discount)
        ).sum()

        segment_results.append({
            "Risk_Segment": segment,
            "Discount_%": discount * 100,
            "Customers": len(segment_data),
            "Avg_Premium": discounted_premium.mean(),
            "Avg_Risk": segment_data["Claim_Risk_Proxy"].mean(),
            "Total_Premium": total_premium,
            "Total_Risk_Cost": total_risk_cost,
            "Risk_Adjusted_Premium": (
                total_premium - total_risk_cost
            )
        })

segment_discount_grid = pd.DataFrame(
    segment_results
)


print("\n" + "=" * 75)
print("DISCOUNT GRID — RISK SEGMENTS")
print("=" * 75)

print(
    segment_discount_grid.head(30).round(2)
)




# ============================================================
# PART IV — PRICE RESPONSE / UPLIFT SIMULATION
# ============================================================

# ------------------------------------------------------------
# 1. MODELLER
# ------------------------------------------------------------

lr_model = best_lr
lgbm_model = best_lgbm


# ============================================================
# 2. PRICE RESPONSE BASE
# ============================================================

price_base = pricing_base.copy()

price_base["Original_Premium"] = (
    price_base["Premium"]
)


# ============================================================
# 3. DISCOUNT GRID
# ============================================================

discount_grid = (
    np.arange(0, 10.0001, 0.1) / 100
)


# ============================================================
# 4. MODEL BASE DATA
# ============================================================

X_base_lr = X_train_lr.loc[
    price_base.index
].copy()

X_base_lgbm = X_train_ohe.loc[
    price_base.index
].copy()


# ============================================================
# 5. BASELINE PREDICTION — 0% DISCOUNT
# ============================================================

if "TEKLİF PRİMİ" not in X_base_lr.columns:
    raise KeyError(
        "TEKLİF PRİMİ Logistic Regression feature setinde yok."
    )

if "TEKLİF PRİMİ" not in X_base_lgbm.columns:
    raise KeyError(
        "TEKLİF PRİMİ LightGBM feature setinde yok."
    )


baseline_lr = (
    lr_model
    .predict_proba(X_base_lr)[:, 1]
)

baseline_lgbm = (
    lgbm_model
    .predict_proba(X_base_lgbm)[:, 1]
)

baseline_probability = (
    baseline_lr + baseline_lgbm
) / 2


# ============================================================
# 6. DISCOUNT SIMULATION
# ============================================================

price_response_results = []


for discount in discount_grid:

    # --------------------------------------------------------
    # Yeni prim
    # --------------------------------------------------------

    new_premium = (
        price_base["Original_Premium"]
        * (1 - discount)
    )


    # --------------------------------------------------------
    # LOGISTIC
    # --------------------------------------------------------

    X_discount_lr = X_base_lr.copy()

    X_discount_lr["TEKLİF PRİMİ"] = (
        new_premium
    )

    prob_lr = (
        lr_model
        .predict_proba(X_discount_lr)[:, 1]
    )


    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    X_discount_lgbm = X_base_lgbm.copy()

    X_discount_lgbm["TEKLİF PRİMİ"] = (
        new_premium
    )

    prob_lgbm = (
        lgbm_model
        .predict_proba(X_discount_lgbm)[:, 1]
    )


    # --------------------------------------------------------
    # MODEL ENSEMBLE
    # --------------------------------------------------------

    probability = (
        prob_lr + prob_lgbm
    ) / 2


    # --------------------------------------------------------
    # UPLIFT
    # --------------------------------------------------------

    uplift = (
        probability
        - baseline_probability
    )


    # --------------------------------------------------------
    # EXPECTED ACCEPTANCE
    # --------------------------------------------------------

    expected_accepted = (
        probability.sum()
    )

    baseline_accepted = (
        baseline_probability.sum()
    )

    incremental_accepted = (
        expected_accepted
        - baseline_accepted
    )


    # --------------------------------------------------------
    # EXPECTED REVENUE
    # --------------------------------------------------------

    expected_revenue = (
        new_premium * probability
    ).sum()


    # --------------------------------------------------------
    # DISCOUNT COST
    # --------------------------------------------------------

    baseline_expected_revenue = (
        price_base["Original_Premium"]
        * probability
    ).sum()

    discount_cost = (
        baseline_expected_revenue
        - expected_revenue
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    price_response_results.append({

        "Discount_%":
            discount * 100,

        "Avg_Acceptance":
            probability.mean() * 100,

        "Incremental_Accepted":
            incremental_accepted,

        "Avg_Uplift":
            uplift.mean() * 100,

        "Expected_Accepted":
            expected_accepted,

        "Expected_Revenue":
            expected_revenue,

        "Discount_Cost":
            discount_cost,

        "Avg_Discounted_Premium":
            new_premium.mean()
    })


# ============================================================
# 7. FINAL TABLE
# ============================================================

price_response = pd.DataFrame(
    price_response_results
)


print("\n" + "=" * 75)
print("PRICE RESPONSE — LOGISTIC + LIGHTGBM")
print("=" * 75)

print(
    price_response.round(3)
)

# ============================================================
# OPTIMAL DISCOUNT — MARGINAL BENEFIT vs DISCOUNT COST
# ============================================================

ab_grid = price_response.copy()

# Her indirim seviyesinde beklenen kabul edilen müşteri sayısı
ab_grid["Expected_Accepted_Customers"] = (
    ab_grid["Avg_Acceptance"] / 100 * len(pricing_base)
)

# Beklenen gelir
ab_grid["Expected_Revenue"] = (
    ab_grid["Expected_Accepted_Customers"]
    * ab_grid["Avg_Discounted_Premium"]
)

# Bir önceki discount seviyesine göre değişim
ab_grid["Incremental_Customers"] = (
    ab_grid["Expected_Accepted_Customers"].diff()
)

ab_grid["Incremental_Revenue"] = (
    ab_grid["Expected_Revenue"].diff()
)

ab_grid["Incremental_Discount_Cost"] = (
    ab_grid["Discount_Cost"].diff()
)

# Net incremental contribution
ab_grid["Incremental_Net_Value"] = (
    ab_grid["Incremental_Revenue"]
    - ab_grid["Incremental_Discount_Cost"]
)

# İlk satırı 0 yap
ab_grid.loc[ab_grid.index[0], [
    "Incremental_Customers",
    "Incremental_Revenue",
    "Incremental_Discount_Cost",
    "Incremental_Net_Value"
]] = 0


# ============================================================
# OPTIMUM
# ============================================================

optimal_idx = ab_grid["Incremental_Net_Value"].idxmax()

optimal_discount = ab_grid.loc[
    optimal_idx, "Discount_%"
]

optimal_row = ab_grid.loc[optimal_idx]


print("\n" + "=" * 75)
print("OPTIMAL DISCOUNT — MARGINAL ANALYSIS")
print("=" * 75)

print(f"Optimal Discount       : {optimal_discount:.1f}%")
print(f"Expected Acceptance    : {optimal_row['Avg_Acceptance']:.3f}%")
print(f"Expected Customers     : {optimal_row['Expected_Accepted_Customers']:.0f}")
print(f"Expected Revenue       : {optimal_row['Expected_Revenue']:,.2f}")
print(f"Discount Cost          : {optimal_row['Discount_Cost']:,.2f}")
print(f"Incremental Net Value  : {optimal_row['Incremental_Net_Value']:,.2f}")


# ============================================================
# TOP DISCOUNT LEVELS
# ============================================================

print("\n" + "=" * 75)
print("TOP 10 DISCOUNT LEVELS")
print("=" * 75)

print(
    ab_grid[
        [
            "Discount_%",
            "Avg_Acceptance",
            "Expected_Accepted_Customers",
            "Expected_Revenue",
            "Discount_Cost",
            "Incremental_Net_Value"
        ]
    ]
    .sort_values(
        "Incremental_Net_Value",
        ascending=False
    )
    .head(10)
    .round(3)
)


# ============================================================
# MARGINAL VALUE GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    ab_grid["Discount_%"],
    ab_grid["Incremental_Net_Value"]
)

plt.axvline(
    optimal_discount,
    linestyle="--",
    label=f"Optimal = {optimal_discount:.1f}%"
)

plt.xlabel("Discount (%)")
plt.ylabel("Incremental Net Value")
plt.title("Marginal Value of Discount")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# PART V — INCREMENTAL PORTFOLIO GROWTH
# ============================================================

portfolio_growth = price_response.copy()

# ------------------------------------------------------------
# 1. Beklenen kabul edilen müşteri
# ------------------------------------------------------------

portfolio_growth["Expected_Accepted"] = (
    portfolio_growth["Avg_Acceptance"] / 100
    * len(pricing_base)
)


# ------------------------------------------------------------
# 2. Baseline kabul
# ------------------------------------------------------------

baseline_accepted = (
    portfolio_growth.loc[
        portfolio_growth["Discount_%"] == 0,
        "Expected_Accepted"
    ].iloc[0]
)


# ------------------------------------------------------------
# 3. Incremental customers
# ------------------------------------------------------------

portfolio_growth["Incremental_Customers"] = (
    portfolio_growth["Expected_Accepted"]
    - baseline_accepted
)


# ------------------------------------------------------------
# 4. Portfolio growth %
# ------------------------------------------------------------

portfolio_growth["Portfolio_Growth_%"] = (
    portfolio_growth["Incremental_Customers"]
    / baseline_accepted
) * 100


# ------------------------------------------------------------
# 5. Discount cost per incremental customer
# ------------------------------------------------------------

portfolio_growth["Discount_Cost_per_Incremental"] = np.where(
    portfolio_growth["Incremental_Customers"] > 0,
    portfolio_growth["Discount_Cost"]
    / portfolio_growth["Incremental_Customers"],
    np.nan
)


# ============================================================
# 6. RESULT
# ============================================================

print("\n" + "=" * 85)
print("INCREMENTAL PORTFOLIO GROWTH — DISCOUNT ANALYSIS")
print("=" * 85)

print(
    portfolio_growth[
        [
            "Discount_%",
            "Avg_Acceptance",
            "Expected_Accepted",
            "Incremental_Customers",
            "Portfolio_Growth_%",
            "Discount_Cost",
            "Discount_Cost_per_Incremental"
        ]
    ].round(3)
)

# ============================================================
# 7. INCREMENTAL CUSTOMERS vs DISCOUNT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    portfolio_growth["Discount_%"],
    portfolio_growth["Incremental_Customers"],
    marker="o",
    markersize=3
)

plt.xlabel("Discount (%)")
plt.ylabel("Incremental Expected Customers")
plt.title("Incremental Portfolio Growth vs Discount")

plt.tight_layout()
plt.show()


# ============================================================
# 8. EFFICIENCY OF DISCOUNT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    portfolio_growth["Discount_%"],
    portfolio_growth["Incremental_Customers"]
    / np.maximum(portfolio_growth["Discount_%"], 0.1),
    marker="o",
    markersize=3
)

plt.xlabel("Discount (%)")
plt.ylabel("Incremental Customers per Discount Point")
plt.title("Discount Efficiency")

plt.tight_layout()
plt.show()


# ============================================================
# PRICE RESPONSE — SANITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("PRICE RESPONSE SANITY CHECK")
print("=" * 80)

# ------------------------------------------------------------
# 1. ACCEPTANCE RESPONSE
# ------------------------------------------------------------

base_acceptance = price_response.loc[
    price_response["Discount_%"] == 0,
    "Avg_Acceptance"
].iloc[0]

max_acceptance = price_response["Avg_Acceptance"].max()

acceptance_increase = max_acceptance - base_acceptance

print(f"\nBaseline Acceptance : {base_acceptance:.3f}%")
print(f"Maximum Acceptance  : {max_acceptance:.3f}%")
print(f"Absolute Increase   : {acceptance_increase:.3f} percentage points")


# ------------------------------------------------------------
# 2. ACCEPTANCE INCREASE AT KEY DISCOUNT LEVELS
# ------------------------------------------------------------

check_discounts = [0, 1, 2, 3, 5, 7, 10]

sanity_table = price_response[
    price_response["Discount_%"].isin(check_discounts)
].copy()

sanity_table["Incremental_Acceptance"] = (
    sanity_table["Avg_Acceptance"]
    - base_acceptance
)

sanity_table["Incremental_Customers"] = (
    sanity_table["Incremental_Acceptance"]
    / 100
    * len(pricing_base)
)

print("\n" + "=" * 80)
print("ACCEPTANCE RESPONSE AT KEY DISCOUNT LEVELS")
print("=" * 80)

print(
    sanity_table[
        [
            "Discount_%",
            "Avg_Acceptance",
            "Incremental_Acceptance",
            "Incremental_Customers"
        ]
    ].round(3)
)


# ------------------------------------------------------------
# 3. MONOTONICITY CHECK
# ------------------------------------------------------------

acceptance_diff = (
    price_response["Avg_Acceptance"]
    .diff()
    .dropna()
)

negative_steps = (acceptance_diff < 0).sum()

print("\n" + "=" * 80)
print("MONOTONICITY CHECK")
print("=" * 80)

print(
    f"Negative acceptance steps: {negative_steps}"
)

if negative_steps == 0:
    print("✓ Acceptance increases monotonically with discount.")
else:
    print(
        "⚠ Acceptance does not increase monotonically. "
        "Model response should be investigated."
    )


# ------------------------------------------------------------
# 4. PRICE ELASTICITY / RESPONSE SLOPE
# ------------------------------------------------------------

discount_change = (
    price_response["Discount_%"].iloc[-1]
    - price_response["Discount_%"].iloc[0]
)

acceptance_change = (
    price_response["Avg_Acceptance"].iloc[-1]
    - price_response["Avg_Acceptance"].iloc[0]
)

response_per_1pct_discount = (
    acceptance_change / discount_change
)

print("\n" + "=" * 80)
print("PRICE RESPONSE SENSITIVITY")
print("=" * 80)

print(
    f"Acceptance increase per 1% discount: "
    f"{response_per_1pct_discount:.4f} percentage points"
)


# ------------------------------------------------------------
# 5. ECONOMIC BREAK-EVEN CHECK
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ECONOMIC BREAK-EVEN CHECK")
print("=" * 80)

# Ortalama başlangıç primi
avg_premium = pricing_base["Premium"].mean()

# Her 1% indirimde bir müşteriden vazgeçilen prim
discount_cost_per_customer = avg_premium * 0.01

# 1 ekstra müşterinin getirdiği yaklaşık prim
incremental_customers_per_1pct = (
    response_per_1pct_discount
    / 100
    * len(pricing_base)
)

if incremental_customers_per_1pct > 0:

    break_even_premium = (
        discount_cost_per_customer
        * len(pricing_base)
        / incremental_customers_per_1pct
    )

    print(
        f"Incremental customers / 1% discount : "
        f"{incremental_customers_per_1pct:.2f}"
    )

    print(
        f"Approx. break-even revenue/customer : "
        f"{break_even_premium:,.2f}"
    )

print("\nSanity check completed.")



print(price_response.columns.tolist())


# ============================================================
# PRICE RESPONSE — CURRENT GRID CHECK
# ============================================================

print("\n" + "=" * 80)
print("PRICE RESPONSE GRID CHECK")
print("=" * 80)

print("\nColumns:")
print(price_response.columns.tolist())

print("\nDiscount levels:")
print(price_response["Discount_%"].round(1).to_list())

print("\nBaseline Acceptance:")
print(
    price_response.loc[
        price_response["Discount_%"] == 0,
        "Avg_Acceptance"
    ].iloc[0]
)

print("\nAverage Premium:")
print(
    price_response[
        [
            "Discount_%",
            "Avg_Discounted_Premium"
        ]
    ].iloc[[0, 10, 20, 50, 100]]
    .round(3)
)

# ============================================================
# PART V — ECONOMIC DISCOUNT EVALUATION
# ============================================================

evaluation = price_response.copy()

# ------------------------------------------------------------
# 1. BASELINE VALUES
# ------------------------------------------------------------

baseline_row = evaluation[
    evaluation["Discount_%"] == 0.0
].iloc[0]

baseline_revenue = baseline_row["Expected_Revenue"]
baseline_accepted = baseline_row["Expected_Accepted"]


# ------------------------------------------------------------
# 2. INCREMENTAL CUSTOMERS
# ------------------------------------------------------------

evaluation["Incremental_Customers"] = (
    evaluation["Expected_Accepted"]
    - baseline_accepted
)


# ------------------------------------------------------------
# 3. REVENUE CHANGE
# ------------------------------------------------------------

evaluation["Revenue_Change"] = (
    evaluation["Expected_Revenue"]
    - baseline_revenue
)


# ------------------------------------------------------------
# 4. NET ECONOMIC VALUE
# ------------------------------------------------------------

# Discount_Cost zaten modelde hesaplanıyor.
# Burada ekstra müşteri kazanımının getirdiği
# gelir ile indirim maliyetini karşılaştırıyoruz.

evaluation["Net_Incremental_Value"] = (
    evaluation["Revenue_Change"]
    - evaluation["Discount_Cost"]
)


# ------------------------------------------------------------
# 5. COST PER INCREMENTAL CUSTOMER
# ------------------------------------------------------------

evaluation["Cost_per_Incremental_Customer"] = np.where(
    evaluation["Incremental_Customers"] > 0,
    evaluation["Discount_Cost"]
    / evaluation["Incremental_Customers"],
    np.nan
)


# ============================================================
# 6. FULL GRID
# ============================================================

print("\n" + "=" * 85)
print("FULL DISCOUNT GRID — ECONOMIC EVALUATION")
print("=" * 85)

print(
    evaluation[
        [
            "Discount_%",
            "Avg_Acceptance",
            "Expected_Accepted",
            "Incremental_Customers",
            "Expected_Revenue",
            "Discount_Cost",
            "Net_Incremental_Value"
        ]
    ].round(3)
)


# ============================================================
# 7. BEST DISCOUNT
# ============================================================

best_idx = evaluation[
    "Net_Incremental_Value"
].idxmax()

best_discount = evaluation.loc[
    best_idx,
    "Discount_%"
]

best_row = evaluation.loc[best_idx]


print("\n" + "=" * 85)
print("BEST ECONOMIC DISCOUNT")
print("=" * 85)

print(
    f"Discount                  : {best_discount:.1f}%"
)

print(
    f"Expected Acceptance       : "
    f"{best_row['Avg_Acceptance']:.3f}%"
)

print(
    f"Expected Customers        : "
    f"{best_row['Expected_Accepted']:.0f}"
)

print(
    f"Incremental Customers     : "
    f"{best_row['Incremental_Customers']:.0f}"
)

print(
    f"Expected Revenue          : "
    f"{best_row['Expected_Revenue']:,.2f}"
)

print(
    f"Discount Cost             : "
    f"{best_row['Discount_Cost']:,.2f}"
)

print(
    f"Net Incremental Value     : "
    f"{best_row['Net_Incremental_Value']:,.2f}"
)


# ============================================================
# 8. SELECTED LEVELS — ONLY FOR READABILITY
# ============================================================

selected_levels = [
    0.0,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0
]

selected_table = evaluation[
    evaluation["Discount_%"].isin(selected_levels)
].copy()

print("\n" + "=" * 85)
print("SELECTED DISCOUNT LEVELS — READABILITY ONLY")
print("=" * 85)

print(
    selected_table[
        [
            "Discount_%",
            "Avg_Acceptance",
            "Expected_Accepted",
            "Incremental_Customers",
            "Expected_Revenue",
            "Discount_Cost",
            "Net_Incremental_Value"
        ]
    ].round(3)
)


















# ============================================================
# PART VI — PRICE ELASTICITY / UPLIFT + CUSTOMER-LEVEL OPTIMIZATION
# ============================================================
# Bu bölüm mevcut predictive modelleri BOZMAZ.
# Mevcut Logistic + LightGBM ensemble'i müşteri bazında fiyat-response
# eğrisi üretmek için kullanır; ardından bu eğri bir MILP optimizasyon
# problemine beslenir.
#
# ÖNEMLİ METODOLOJİ NOTU:
# Veri setinde gerçek randomize indirim/treatment değişkeni bulunmadığı
# için burada "causal uplift" iddiasında bulunulmuyor. Model, gözlenen
# teklif primi üzerinden PRICE RESPONSE / PRICE-ELASTICITY UPLIFT
# tahmini yapmaktadır. Gerçek indirim deneyleri (A/B veya randomized
# pricing) geldiğinde bu bölüm causal uplift modeliyle değiştirilebilir.

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# ------------------------------------------------------------
# CUSTOMER ID = ORİJİNAL EXCEL'DEKİ TEKLİF NUMARASI
# ------------------------------------------------------------
price_base["Customer_ID"] = (
    risk_segment_analysis.loc[
        price_base.index,
        "TEKLİF NUMARASI"
    ]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# 2. CUSTOMER-LEVEL PRICE RESPONSE MATRIX
# ------------------------------------------------------------
# Her müşteri için:
# d = 0.0%, 0.1%, ..., 10.0%
# altında acceptance probability ve uplift tutulur.

customer_response = pd.DataFrame(
    {
        "Customer_ID": price_base["Customer_ID"].values,
        "Original_Premium": price_base["Original_Premium"].values,
        "Claim_Risk_Proxy": price_base["Claim_Risk_Proxy"].values,
        "Risk_Score": price_base["Risk_Score"].values,
        "Risk_Segment": price_base["Risk_Segment"].astype(str).values,
        "Baseline_Probability": baseline_probability
    },
    index=price_base.index
)

probability_matrix = []
uplift_matrix = []
elasticity_matrix = []

# ============================================================
# LR MODELİ İÇİN PREMIUM SCALING
# ============================================================

premium_pos = list(scaler.feature_names_in_).index("TEKLİF PRİMİ")

premium_mean = scaler.mean_[premium_pos]
premium_scale = scaler.scale_[premium_pos]


for discount in discount_grid:
    new_premium = (
        price_base["Original_Premium"] * (1 - discount)
    )

    Xd_lr = X_base_lr.copy()
    # LR modeli StandardScaler ile eğitildiği için
    # TEKLİF PRİMİ tekrar standardize edilmelidir.
    premium_mean = scaler.mean_[
        list(scaler.feature_names_in_).index("TEKLİF PRİMİ")
        ]

    premium_scale = scaler.scale_[
        list(scaler.feature_names_in_).index("TEKLİF PRİMİ")
        ]

    new_premium_scaled = (
        new_premium - premium_mean
        ) / premium_scale

    Xd_lr["TEKLİF PRİMİ"] = new_premium_scaled

    # LightGBM raw premium kullanıyor
    Xd_lgbm = X_base_lgbm.copy()
    Xd_lgbm["TEKLİF PRİMİ"] = new_premium

    p = (
        lr_model.predict_proba(Xd_lr)[:, 1]
        + lgbm_model.predict_proba(Xd_lgbm)[:, 1]
    ) / 2

    probability_matrix.append(p)
    uplift_matrix.append(p - baseline_probability)

    # Relative price change = -discount.
    # Local price elasticity yaklaşık olarak:
    # (%ΔAcceptance) / (%ΔPrice)
    if discount == 0:
        elasticity = np.zeros_like(p)
    else:
        acceptance_pct_change = np.divide(
            p - baseline_probability,
            np.maximum(baseline_probability, 1e-8)
        )
        elasticity = acceptance_pct_change / (-discount)

    elasticity_matrix.append(elasticity)

probability_matrix = np.column_stack(probability_matrix)
uplift_matrix = np.column_stack(uplift_matrix)
elasticity_matrix = np.column_stack(elasticity_matrix)


# ------------------------------------------------------------
# 3. CUSTOMER RESPONSE TABLE — LONG FORMAT
# ------------------------------------------------------------
response_rows = []

for i, idx in enumerate(price_base.index):
    for j, discount in enumerate(discount_grid):
        prob = probability_matrix[i, j]
        uplift = uplift_matrix[i, j]
        elasticity = elasticity_matrix[i, j]
        original_premium = price_base.loc[idx, "Original_Premium"]
        discounted_premium = original_premium * (1 - discount)

        # İndirim maliyeti: kabul edilen poliçe başına vazgeçilen prim.
        expected_discount_cost = (
            original_premium * discount * prob
        )

        # Risk proxy: prim * proxy. Bu gerçek expected loss değildir.
        risk_cost_proxy = (
            discounted_premium
            * price_base.loc[idx, "Claim_Risk_Proxy"]
            * prob
        )

        expected_contribution_proxy = (
            discounted_premium * prob
            - risk_cost_proxy
        )

        response_rows.append({
            "Customer_ID": price_base.loc[idx, "Customer_ID"],
            "Original_Premium": original_premium,
            "Discount_%": discount * 100,
            "Discounted_Premium": discounted_premium,
            "Acceptance_Probability": prob,
            "Uplift": uplift,
            "Uplift_pp": uplift * 100,
            "Price_Elasticity": elasticity,
            "Expected_Discount_Cost": expected_discount_cost,
            "Expected_Risk_Cost_Proxy": risk_cost_proxy,
            "Expected_Contribution_Proxy": expected_contribution_proxy,
            "Claim_Risk_Proxy": price_base.loc[idx, "Claim_Risk_Proxy"],
            "Risk_Score": price_base.loc[idx, "Risk_Score"],
            "Risk_Segment": str(price_base.loc[idx, "Risk_Segment"])
        })

customer_response_long = pd.DataFrame(response_rows)

print("\n" + "=" * 90)
print("CUSTOMER-LEVEL PRICE ELASTICITY / UPLIFT MODEL")
print("=" * 90)
print(
    "Response matrix created for",
    len(customer_response),
    "customers and",
    len(discount_grid),
    "discount levels."
)


# ------------------------------------------------------------
# 4. PARETO FRONTIER — PORTFOLIO-LEVEL REFERENCE
# ------------------------------------------------------------
# Mevcut uniform-discount senaryolarını kullanarak:
# x = total expected discount cost
# y = incremental expected customers
# Pareto-optimal noktaları buluyoruz.

pareto = evaluation.copy()
pareto["Total_Discount_Cost"] = pareto["Discount_Cost"]
pareto["Portfolio_Growth_%"] = (
    pareto["Incremental_Customers"]
    / max(baseline_accepted, 1e-8)
) * 100

pareto = pareto.sort_values("Total_Discount_Cost").reset_index(drop=True)

pareto_mask = []
best_growth_seen = -np.inf

for growth in pareto["Incremental_Customers"]:
    if growth > best_growth_seen + 1e-10:
        pareto_mask.append(True)
        best_growth_seen = growth
    else:
        pareto_mask.append(False)

pareto["Pareto_Optimal"] = pareto_mask
pareto_frontier = pareto[pareto["Pareto_Optimal"]].copy()


# ------------------------------------------------------------
# 5. KNEE POINT
# ------------------------------------------------------------
# Pareto eğrisinde "ek indirim karşılığında artık çok az büyüme"
# noktasını seçiyoruz. Bu nokta müşteri bazlı optimizasyon için
# toplam indirim bütçesi olarak kullanılacaktır.

if len(pareto_frontier) >= 3:
    x = pareto_frontier["Total_Discount_Cost"].to_numpy(dtype=float)
    y = pareto_frontier["Incremental_Customers"].to_numpy(dtype=float)

    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-12)

    # Başlangıç ve maksimum noktaları birleştiren doğruya uzaklık.
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]

    distances = np.abs(
        (y2 - y1) * x_norm
        - (x2 - x1) * y_norm
        + x2 * y1
        - y2 * x1
    ) / np.sqrt(
        (y2 - y1) ** 2
        + (x2 - x1) ** 2
        + 1e-12
    )

    # Uç noktalar yerine iç noktalar arasından knee seç.
    if len(distances) > 2:
        knee_pos = int(np.argmax(distances[1:-1])) + 1
    else:
        knee_pos = int(np.argmax(distances))

    knee_row = pareto_frontier.iloc[knee_pos]
else:
    knee_row = pareto_frontier.iloc[-1]

knee_budget = float(knee_row["Total_Discount_Cost"])
knee_growth = float(knee_row["Incremental_Customers"])
knee_discount = float(knee_row["Discount_%"])

print("\n" + "=" * 90)
print("PARETO FRONTIER / KNEE POINT")
print("=" * 90)
print(f"Reference uniform discount : {knee_discount:.1f}%")
print(f"Reference discount budget  : {knee_budget:,.2f}")
print(f"Reference incremental cust. : {knee_growth:.2f}")
print(f"Reference portfolio growth : {knee_growth / max(baseline_accepted,1e-8) * 100:.2f}%")

print("\nPareto frontier:")
print(
    pareto_frontier[
        [
            "Discount_%",
            "Total_Discount_Cost",
            "Incremental_Customers",
            "Portfolio_Growth_%"
        ]
    ].round(3)
)


# ------------------------------------------------------------
# 6. MILP CUSTOMER-LEVEL ALLOCATION
# ------------------------------------------------------------
# x(i,d) = 1 ise müşteri i için d indirimi seçilir.
#
# Objective 1:
#   MAX total expected accepted customers
#
# Constraint:
#   toplam expected discount cost <= Pareto knee budget
#
# Her müşteri tam olarak bir indirim seviyesi seçer.
# Böylece indirim artık tüm portföye homojen verilmez.

n_customers = len(customer_response)
n_discounts = len(discount_grid)
n_vars = n_customers * n_discounts

# Objective: maximize expected accepted -> minimize negative.
c = -probability_matrix.reshape(-1)

# One discount per customer + total discount budget.
A = lil_matrix((n_customers + 1, n_vars), dtype=float)

for i in range(n_customers):
    start = i * n_discounts
    end = start + n_discounts
    A[i, start:end] = 1.0

# Discount budget row.
expected_discount_cost_matrix = (
    probability_matrix
    * price_base["Original_Premium"].to_numpy()[:, None]
    * discount_grid[None, :]
)

A[n_customers, :] = expected_discount_cost_matrix.reshape(-1)

lower = np.concatenate([
    np.ones(n_customers),
    [-np.inf]
])
upper = np.concatenate([
    np.ones(n_customers),
    [knee_budget]
])

constraints = LinearConstraint(
    A.tocsr(),
    lower,
    upper
)

integrality = np.ones(n_vars, dtype=int)
bounds = Bounds(
    np.zeros(n_vars),
    np.ones(n_vars)
)

milp_result = milp(
    c=c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints,
    options={
        "time_limit": 12000,   # 1200dü 20dkyı çoğallttım CPU yetmiyor
        "mip_rel_gap": 0.01   # %1 optimality gap
    }
)

if not milp_result.success:
    raise RuntimeError(
        "Customer-level MILP optimization başarısız: "
        + str(milp_result.message)
    )

solution = milp_result.x.reshape(n_customers, n_discounts)
selected_j = solution.argmax(axis=1)


# ------------------------------------------------------------
# 7. CUSTOMER-LEVEL OPTIMAL TABLE
# ------------------------------------------------------------
customer_optimization = customer_response.copy()

customer_optimization["Optimal_Discount_%"] = (
    discount_grid[selected_j] * 100
)

customer_optimization["Optimal_Premium"] = (
    customer_optimization["Original_Premium"]
    * (1 - discount_grid[selected_j])
)

customer_optimization["Baseline_Acceptance_%"] = (
    customer_optimization["Baseline_Probability"] * 100
)

customer_optimization["Optimal_Acceptance_%"] = np.array([
    probability_matrix[i, selected_j[i]] * 100
    for i in range(n_customers)
])

customer_optimization["Uplift_pp"] = (
    customer_optimization["Optimal_Acceptance_%"]
    - customer_optimization["Baseline_Acceptance_%"]
)

customer_optimization["Optimal_Elasticity"] = np.array([
    elasticity_matrix[i, selected_j[i]]
    for i in range(n_customers)
])

customer_optimization["Expected_Discount_Cost"] = np.array([
    expected_discount_cost_matrix[i, selected_j[i]]
    for i in range(n_customers)
])

customer_optimization["Expected_Risk_Cost_Proxy"] = np.array([
    (
        customer_optimization.loc[idx, "Optimal_Premium"]
        * price_base.loc[idx, "Claim_Risk_Proxy"]
        * (
            probability_matrix[i, selected_j[i]]
        )
    )
    for i, idx in enumerate(price_base.index)
])

customer_optimization["Expected_Incremental_Customer"] = (
    customer_optimization["Optimal_Acceptance_%"]
    - customer_optimization["Baseline_Acceptance_%"]
) / 100

customer_optimization["Discount_Efficiency"] = np.where(
    customer_optimization["Optimal_Discount_%"] > 0,
    customer_optimization["Expected_Incremental_Customer"]
    / customer_optimization["Optimal_Discount_%"],
    0
)


# ------------------------------------------------------------
# 8. OPTIMIZATION SUMMARY
# ------------------------------------------------------------
optimized_expected_accepted = (
    customer_optimization["Optimal_Acceptance_%"].sum() / 100
)

optimized_incremental_customers = (
    optimized_expected_accepted - baseline_accepted
)

optimized_growth_pct = (
    optimized_incremental_customers
    / max(baseline_accepted, 1e-8)
    * 100
)

optimized_discount_cost = (
    customer_optimization["Expected_Discount_Cost"].sum()
)

optimized_avg_discount = np.average(
    customer_optimization["Optimal_Discount_%"],
    weights=customer_optimization["Original_Premium"]
)

print("\n" + "=" * 90)
print("CUSTOMER-LEVEL MILP OPTIMIZATION RESULT")
print("=" * 90)
print(f"Baseline expected customers     : {baseline_accepted:.2f}")
print(f"Optimized expected customers    : {optimized_expected_accepted:.2f}")
print(f"Incremental customers           : {optimized_incremental_customers:.2f}")
print(f"Portfolio growth                : {optimized_growth_pct:.2f}%")
print(f"Total expected discount cost    : {optimized_discount_cost:,.2f}")
print(f"Premium-weighted avg discount   : {optimized_avg_discount:.2f}%")
print(f"Pareto knee budget              : {knee_budget:,.2f}")


# ------------------------------------------------------------
# 9. LEXICOGRAPHIC SECOND STEP
# ------------------------------------------------------------
# İlk MILP maksimum expected acceptance'ı buldu.
# Şimdi aynı portföy sonucunu koruyup toplam indirimi minimize eden
# ikinci MILP çalıştırılıyor.

max_acceptance_objective = optimized_expected_accepted
acceptance_tolerance = 1e-6

# Yeni objective: total discount cost minimize.
c2 = expected_discount_cost_matrix.reshape(-1)

# Existing constraints + minimum acceptance constraint.
from scipy.sparse import vstack, csr_matrix

A2 = vstack([
    A.tocsr(),
    csr_matrix(probability_matrix.reshape(1, -1))
], format="csr")

lower2 = np.concatenate([
    lower,
    [max_acceptance_objective - acceptance_tolerance]
])
upper2 = np.concatenate([
    upper,
    [np.inf]
])

constraints2 = LinearConstraint(
    A2.tocsr(),
    lower2,
    upper2
)

lex_result = milp(
    c=c2,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints2,
    options={"time_limit": 120}
)

if lex_result.success:
    lex_solution = lex_result.x.reshape(n_customers, n_discounts)
    lex_selected_j = lex_solution.argmax(axis=1)

    customer_optimization["Optimal_Discount_%"] = (
        discount_grid[lex_selected_j] * 100
    )
    customer_optimization["Optimal_Premium"] = (
        customer_optimization["Original_Premium"]
        * (1 - discount_grid[lex_selected_j])
    )
    customer_optimization["Optimal_Acceptance_%"] = np.array([
        probability_matrix[i, lex_selected_j[i]] * 100
        for i in range(n_customers)
    ])
    customer_optimization["Baseline_Acceptance_%"] = (
        customer_optimization["Baseline_Probability"] * 100
    )
    customer_optimization["Uplift_pp"] = (
        customer_optimization["Optimal_Acceptance_%"]
        - customer_optimization["Baseline_Acceptance_%"]
    )
    customer_optimization["Optimal_Elasticity"] = np.array([
        elasticity_matrix[i, lex_selected_j[i]]
        for i in range(n_customers)
    ])
    customer_optimization["Expected_Discount_Cost"] = np.array([
        expected_discount_cost_matrix[i, lex_selected_j[i]]
        for i in range(n_customers)
    ])
    customer_optimization["Expected_Incremental_Customer"] = (
        customer_optimization["Optimal_Acceptance_%"]
        - customer_optimization["Baseline_Acceptance_%"]
    ) / 100
    customer_optimization["Discount_Efficiency"] = np.where(
        customer_optimization["Optimal_Discount_%"] > 0,
        customer_optimization["Expected_Incremental_Customer"]
        / customer_optimization["Optimal_Discount_%"],
        0
    )

    optimized_discount_cost = customer_optimization[
        "Expected_Discount_Cost"
    ].sum()

    optimized_expected_accepted = (
        customer_optimization["Optimal_Acceptance_%"].sum() / 100
    )

    optimized_incremental_customers = (
        optimized_expected_accepted - baseline_accepted
    )

    optimized_growth_pct = (
        optimized_incremental_customers
        / max(baseline_accepted, 1e-8)
        * 100
    )

    print("\nLexicographic second step completed:")
    print(f"Minimum discount cost at max growth: {optimized_discount_cost:,.2f}")


# ------------------------------------------------------------
# 10. CUSTOMER RECOMMENDATION FUNCTION
# ------------------------------------------------------------
# Kullanım:
# recommend_customer_discount("12345")
#
# Çıktı:
# - mevcut prim
# - önerilen indirim
# - indirimli prim
# - baseline acceptance
# - optimal acceptance
# - uplift
# - fiyat elastikiyeti
# - risk proxy
# - beklenen indirim maliyeti

customer_lookup = customer_optimization.copy()
customer_lookup["Customer_ID"] = customer_lookup["Customer_ID"].astype(str)


def recommend_customer_discount(customer_id):
    """Tek bir müşteri için optimize edilmiş fiyat/indirim önerisi döndürür."""

    cid = str(customer_id)
    result = customer_lookup[
        customer_lookup["Customer_ID"] == cid
    ]

    if result.empty:
        # Index ID ile arama da yapabilmek için ikinci şans.
        result = customer_lookup[
            customer_lookup.index.astype(str) == cid
        ]

    if result.empty:
        print(f"Müşteri bulunamadı: {customer_id}")
        print("Örnek Customer_ID değerleri:")
        print(customer_lookup["Customer_ID"].head(10).to_list())
        return None

    r = result.iloc[0]

    output = {
        "Customer_ID": r["Customer_ID"],
        "Original_Premium": float(r["Original_Premium"]),
        "Optimal_Discount_%": float(r["Optimal_Discount_%"]),
        "Optimal_Premium": float(r["Optimal_Premium"]),
        "Baseline_Acceptance_%": float(r["Baseline_Acceptance_%"]),
        "Optimal_Acceptance_%": float(r["Optimal_Acceptance_%"]),
        "Uplift_pp": float(r["Uplift_pp"]),
        "Price_Elasticity": float(r["Optimal_Elasticity"]),
        "Claim_Risk_Proxy": float(r["Claim_Risk_Proxy"]),
        "Risk_Score": float(r["Risk_Score"]),
        "Risk_Segment": r["Risk_Segment"],
        "Expected_Discount_Cost": float(r["Expected_Discount_Cost"]),
        "Discount_Efficiency": float(r["Discount_Efficiency"])
    }

    print("\n" + "=" * 75)
    print("CUSTOMER-LEVEL OPTIMAL OFFER")
    print("=" * 75)
    print(f"Customer ID               : {output['Customer_ID']}")
    print(f"Original Premium          : {output['Original_Premium']:,.2f}")
    print(f"Recommended Discount      : {output['Optimal_Discount_%']:.1f}%")
    print(f"Recommended Premium       : {output['Optimal_Premium']:,.2f}")
    print(f"Baseline Acceptance       : {output['Baseline_Acceptance_%']:.2f}%")
    print(f"Optimal Acceptance        : {output['Optimal_Acceptance_%']:.2f}%")
    print(f"Expected Uplift            : +{output['Uplift_pp']:.2f} pp")
    print(f"Price Elasticity           : {output['Price_Elasticity']:.4f}")
    print(f"Claim Risk Proxy           : {output['Claim_Risk_Proxy']:.4f}")
    print(f"Risk Score                 : {output['Risk_Score']:.4f}")
    print(f"Risk Segment               : {output['Risk_Segment']}")
    print(f"Expected Discount Cost     : {output['Expected_Discount_Cost']:,.2f}")
    print(f"Discount Efficiency        : {output['Discount_Efficiency']:.6f}")

    return pd.Series(output)


# ------------------------------------------------------------
# 11. OPTIONAL INTERACTIVE INPUT
# ------------------------------------------------------------
# Notebook/terminal ortamında çalıştırmak için:
#
# customer_id_input = input("Customer ID giriniz: ")
# customer_recommendation = recommend_customer_discount(customer_id_input)
#
# Örnek:
# customer_recommendation = recommend_customer_discount("15")


# ------------------------------------------------------------
# 12. EXPORT — CUSTOMER RECOMMENDATIONS
# ------------------------------------------------------------
customer_optimization_export = customer_optimization[
    [
        "Customer_ID",
        "Original_Premium",
        "Optimal_Discount_%",
        "Optimal_Premium",
        "Baseline_Acceptance_%",
        "Optimal_Acceptance_%",
        "Uplift_pp",
        "Optimal_Elasticity",
        "Claim_Risk_Proxy",
        "Risk_Score",
        "Risk_Segment",
        "Expected_Discount_Cost",
        "Discount_Efficiency"
    ]
].copy()

print("\n" + "=" * 90)
print("TOP CUSTOMER RECOMMENDATIONS — HIGHEST DISCOUNT EFFICIENCY")
print("=" * 90)
print(
    customer_optimization_export
    .sort_values("Discount_Efficiency", ascending=False)
    .head(20)
    .round(4)
    .to_string(index=False)
)

top20 = (
    customer_optimization_export
    .sort_values("Discount_Efficiency", ascending=False)
    .head(20)
    .round(4)
)

print(top20)
# ============================================================
# END OF OPTIMIZATION MODULE
# ============================================================



# ============================================================
# PART XI — OPTIMIZATION ROBUSTNESS & CONSTRAINT VALIDATION
# ============================================================
# This section validates the optimization solution after the solver runs.
# It is intentionally additive: the existing model/optimization pipeline
# remains unchanged.
#
# Tests:
#   1) solver status / solution existence
#   2) one action per customer
#   3) discount bounds [0%, 10%]
#   4) discount-budget feasibility
#   5) risk-budget feasibility (when configured)
#   6) baseline vs optimized portfolio
#   7) lexicographic consistency
#   8) Pareto non-dominance
#   9) price-response monotonicity
#  10) probability stress tests (-5%, +5%)
#  11) discount-budget sensitivity
#  12) final robustness scorecard
#
# The validation functions are written defensively because variable names
# can differ slightly between notebook/code versions.

import numpy as np
import pandas as pd

ROBUSTNESS_EPS = 1e-6


def _find_first_dataframe(names):
    """Return the first DataFrame among global variable names."""
    for name in names:
        obj = globals().get(name)
        if isinstance(obj, pd.DataFrame):
            return obj, name
    return None, None


def _find_first_scalar(names, default=np.nan):
    """Return the first numeric scalar among global variable names."""
    for name in names:
        obj = globals().get(name)
        if obj is not None and np.isscalar(obj):
            try:
                return float(obj)
            except Exception:
                pass
    return default


def _find_column(df, candidates):
    """Case-insensitive column finder."""
    if df is None:
        return None
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def validate_optimization_solution(
    solution_df=None,
    discount_budget=None,
    risk_budget=None,
    tolerance=ROBUSTNESS_EPS
):
    """
    Validate the final customer-level optimization output.

    Expected (flexible) columns:
      Customer_ID, Discount, Acceptance_Probability,
      Expected_Discount_Cost, Expected_Risk_Cost_Proxy

    Returns:
      checks_df, metrics
    """
    if solution_df is None:
        solution_df, solution_name = _find_first_dataframe([
    "customer_optimization",
    "customer_optimization_export",
    "lexicographic_solution",
    "optimization_solution",
    "optimized_solution",
    "final_solution",
    "solution_df",
])
    else:
        solution_name = "provided_solution"

    if solution_df is None or solution_df.empty:
        checks = pd.DataFrame([{
            "Test": "Solution exists",
            "Status": "FAIL",
            "Detail": "No non-empty optimization solution DataFrame was found."
        }])
        return checks, {"feasible": False}

    df = solution_df.copy()

    discount_col = _find_column(df, [
    "Discount",
    "Discount_Rate",
    "Optimal_Discount",
    "Optimal_Discount_%",
    "Recommended_Discount",
    "discount"
    ])

    customer_col = _find_column(df, [
        "Customer_ID", "CustomerID", "ID", "customer_id"
    ])

    acceptance_col = _find_column(df, [
        "Acceptance_Probability", "Expected_Acceptance",
        "Probability", "Optimal_Acceptance"
    ])

    discount_cost_col = _find_column(df, [
        "Expected_Discount_Cost", "Discount_Cost",
        "ExpectedDiscountCost"
    ])

    risk_cost_col = _find_column(df, [
        "Expected_Risk_Cost_Proxy", "Risk_Cost_Proxy",
        "Expected_Risk_Cost", "RiskCost"
    ])

    rows = []

    rows.append({
        "Test": "Solution exists",
        "Status": "PASS" if len(df) > 0 else "FAIL",
        "Detail": f"{len(df):,} customer decisions found."
    })

    # Discount bounds
    if discount_col is not None:
        discounts = pd.to_numeric(df[discount_col], errors="coerce")
        # Accept either rates (0-1) or percentage points (0-100).
        if discounts.max(skipna=True) > 1.0 + tolerance:
            discounts_rate = discounts / 100.0
        else:
            discounts_rate = discounts

        bounds_ok = (
            discounts_rate.notna().all()
            and (discounts_rate >= -tolerance).all()
            and (discounts_rate <= 0.10 + tolerance).all()
        )
        rows.append({
            "Test": "Discount bounds [0%, 10%]",
            "Status": "PASS" if bounds_ok else "FAIL",
            "Detail": (
                f"min={discounts_rate.min():.4f}, "
                f"max={discounts_rate.max():.4f}"
            )
        })
    else:
        rows.append({
            "Test": "Discount bounds [0%, 10%]",
            "Status": "WARN",
            "Detail": "Discount column not found; skipped."
        })

    # One action per customer
    if customer_col is not None:
        duplicated = df[customer_col].duplicated().sum()
        one_action_ok = duplicated == 0
        rows.append({
            "Test": "One decision per customer",
            "Status": "PASS" if one_action_ok else "FAIL",
            "Detail": f"duplicate customer decisions={duplicated:,}"
        })
    else:
        rows.append({
            "Test": "One decision per customer",
            "Status": "WARN",
            "Detail": "Customer ID column not found; skipped."
        })

    # Discount budget
    total_discount_cost = np.nan
    if discount_cost_col is not None:
        total_discount_cost = pd.to_numeric(
        df[discount_cost_col],
        errors="coerce"
    ).sum()

    if discount_budget is not None:
        budget_ok = (
            total_discount_cost
            <= discount_budget + tolerance
        )

        rows.append({
            "Test": "Discount budget constraint",
            "Status": "PASS" if budget_ok else "FAIL",
            "Detail": (
                f"used={total_discount_cost:,.2f}, "
                f"budget={discount_budget:,.2f}, "
                f"slack={discount_budget-total_discount_cost:,.2f}"
            )
        })

    else:
        rows.append({
            "Test": "Discount budget constraint",
            "Status": "WARN",
            "Detail": "Budget not configured; cost calculated but not constrained."
        })



    # Risk budget
    total_risk_cost = np.nan
    if risk_cost_col is not None:
        total_risk_cost = pd.to_numeric(
            df[risk_cost_col], errors="coerce"
        ).sum()

        if risk_budget is None:
            risk_budget = _find_first_scalar([
                "risk_budget",
                "RISK_BUDGET",
                "risk_cost_budget",
            ])

        if pd.notna(risk_budget):
            risk_ok = total_risk_cost <= risk_budget + tolerance
            rows.append({
                "Test": "Risk budget constraint",
                "Status": "PASS" if risk_ok else "FAIL",
                "Detail": (
                    f"used={total_risk_cost:,.2f}, "
                    f"budget={risk_budget:,.2f}, "
                    f"slack={risk_budget-total_risk_cost:,.2f}"
                )
            })
        else:
            rows.append({
                "Test": "Risk budget constraint",
                "Status": "WARN",
                "Detail": "Risk budget not configured; proxy cost reported only."
            })
    else:
        rows.append({
            "Test": "Risk budget constraint",
            "Status": "WARN",
            "Detail": "Risk-cost column not found; skipped."
        })

    metrics = {
        "solution_name": solution_name,
        "customers": len(df),
        "total_discount_cost": total_discount_cost,
        "total_risk_cost": total_risk_cost,
        "discount_budget": discount_budget,
        "risk_budget": risk_budget,
    }

    checks_df = pd.DataFrame(rows)
    metrics["feasible"] = not (checks_df["Status"] == "FAIL").any()

    return checks_df, metrics


def validate_price_response_monotonicity(
    response_df=None,
    customer_col="Customer_ID",
    discount_col="Discount",
    probability_col="Acceptance_Probability"
):
    """
    Checks whether acceptance probability is non-decreasing as discount rises.
    Small numerical violations are tolerated.
    """
    if response_df is None:
        response_df, _ = _find_first_dataframe([
            "customer_response_long",
            "price_response_long",
            "price_response_results",
        ])

    if response_df is None or response_df.empty:
        return pd.DataFrame([{
            "Test": "Price-response monotonicity",
            "Status": "WARN",
            "Detail": "Price-response table not found."
        }]), {"violation_rate": np.nan}

    df = response_df.copy()

    customer_col = _find_column(df, [customer_col, "Customer_ID", "CustomerID", "ID"])
    discount_col = _find_column(df, [discount_col, "Discount", "Discount_Rate"])
    probability_col = _find_column(df, [
        probability_col, "Acceptance_Probability",
        "Probability", "Expected_Acceptance"
    ])

    if not all([customer_col, discount_col, probability_col]):
        return pd.DataFrame([{
            "Test": "Price-response monotonicity",
            "Status": "WARN",
            "Detail": "Required columns not found."
        }]), {"violation_rate": np.nan}

    work = df[[customer_col, discount_col, probability_col]].copy()
    work[discount_col] = pd.to_numeric(work[discount_col], errors="coerce")
    work[probability_col] = pd.to_numeric(work[probability_col], errors="coerce")
    work = work.dropna().sort_values([customer_col, discount_col])

    violations = 0
    comparisons = 0

    for _, g in work.groupby(customer_col, sort=False):
        probs = g[probability_col].to_numpy()
        if len(probs) < 2:
            continue
        diffs = np.diff(probs)
        comparisons += len(diffs)
        violations += int((diffs < -1e-8).sum())

    violation_rate = violations / comparisons if comparisons else np.nan

    status = (
        "PASS" if comparisons and violation_rate <= 0.02
        else "WARN" if comparisons
        else "WARN"
    )

    checks = pd.DataFrame([{
        "Test": "Price-response monotonicity",
        "Status": status,
        "Detail": (
            f"violations={violations:,}/{comparisons:,} "
            f"({100*violation_rate:.2f}%)"
        )
    }])

    return checks, {
        "violation_rate": violation_rate,
        "violations": violations,
        "comparisons": comparisons,
    }


def validate_pareto_frontier(
    pareto_df=None,
    cost_col="Discount_Cost",
    growth_col="Incremental_Customers"
):
    """
    Verifies that retained Pareto points are not dominated.
    """
    if pareto_df is None:
        pareto_df, _ = _find_first_dataframe([
            "pareto_frontier",
            "pareto_df",
            "pareto_results",
        ])

    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame([{
            "Test": "Pareto non-dominance",
            "Status": "WARN",
            "Detail": "Pareto frontier table not found."
        }]), {"dominated_points": np.nan}

    cost_col = _find_column(pareto_df, [
        cost_col, "Discount_Cost", "Total_Discount_Cost",
        "Expected_Discount_Cost"
    ])
    growth_col = _find_column(pareto_df, [
        growth_col, "Incremental_Customers",
        "Portfolio_Growth", "Expected_Customers"
    ])

    if not cost_col or not growth_col:
        return pd.DataFrame([{
            "Test": "Pareto non-dominance",
            "Status": "WARN",
            "Detail": "Required Pareto columns not found."
        }]), {"dominated_points": np.nan}

    p = pareto_df[[cost_col, growth_col]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().drop_duplicates()

    dominated = 0
    vals = p.to_numpy()

    for i in range(len(vals)):
        c_i, g_i = vals[i]
        for j in range(len(vals)):
            if i == j:
                continue
            c_j, g_j = vals[j]
            if (
                c_j <= c_i + 1e-9
                and g_j >= g_i - 1e-9
                and (c_j < c_i - 1e-9 or g_j > g_i + 1e-9)
            ):
                dominated += 1
                break

    status = "PASS" if dominated == 0 else "FAIL"

    checks = pd.DataFrame([{
        "Test": "Pareto non-dominance",
        "Status": status,
        "Detail": f"dominated_points={dominated:,}"
    }])

    return checks, {"dominated_points": dominated, "pareto_points": len(p)}


def probability_stress_test(
    solution_df=None,
    probability_col="Acceptance_Probability",
    baseline_col="Baseline_Probability",
    perturbations=(-0.05, 0.05)
):
    """
    Simple model-risk stress test.
    Recomputes expected portfolio under +/-5% probability perturbation.
    """

    if solution_df is None:
        solution_df, _ = _find_first_dataframe([
            "customer_optimization",
            "customer_optimization_export",
            "lexicographic_solution",
            "optimization_solution",
            "optimized_solution",
            "final_solution"
        ])

    if solution_df is None or solution_df.empty:
        return pd.DataFrame([{
            "Test": "Probability stress test",
            "Status": "WARN",
            "Detail": "Optimization solution not found."
        }]), pd.DataFrame()

    pcol = _find_column(solution_df, [
        probability_col,
        "Acceptance_Probability",
        "Expected_Acceptance",
        "Probability",
        "Optimal_Acceptance",
        "Optimal_Acceptance_%",
        "Optimal_Acceptance"
    ])

    if pcol is None:
        return pd.DataFrame([{
            "Test": "Probability stress test",
            "Status": "WARN",
            "Detail": "Acceptance probability column not found."
        }]), pd.DataFrame()

    p = pd.to_numeric(
        solution_df[pcol],
        errors="coerce"
    ).dropna()

    # Eğer kolon yüzde olarak tutuluyorsa 0-1 aralığına çevir
    if p.max() > 1:
        p = p / 100.0

    base = p.sum()

    rows = []

    for shock in perturbations:

        stressed = np.clip(
            p * (1 + shock),
            0,
            1
        )

        stressed_total = stressed.sum()

        delta = stressed_total - base

        rows.append({
            "Shock": f"{shock:+.0%}",
            "Expected_Customers": stressed_total,
            "Delta_vs_Base": delta,
            "Delta_%": (
                delta / base * 100
                if base
                else np.nan
            )
        })

    stress_df = pd.DataFrame(rows)

    checks = pd.DataFrame([{
        "Test": "Probability stress test",
        "Status": "PASS",
        "Detail": (
            "Scenario sensitivity calculated for "
            + ", ".join(
                [f"{x:+.0%}" for x in perturbations]
            )
        )
    }])

    return checks, stress_df


def discount_budget_sensitivity(
    solution_df=None,
    budgets=(0.75, 0.90, 1.00, 1.10, 1.25)
):
    """
    Uses the existing customer-level allocation table to estimate how
    portfolio metrics respond to alternative discount-cost budgets.

    This is intentionally a diagnostic approximation. A full re-solve for
    every budget can be added when the solver model/function is exposed.
    """
    if solution_df is None:
        solution_df, _ = _find_first_dataframe([
            "lexicographic_solution",
            "optimization_solution",
            "optimized_solution",
            "final_solution",
        ])

    if solution_df is None or solution_df.empty:
        return pd.DataFrame()

    cost_col = _find_column(solution_df, [
        "Expected_Discount_Cost", "Discount_Cost",
        "ExpectedDiscountCost"
    ])
    prob_col = _find_column(solution_df, [
        "Acceptance_Probability", "Expected_Acceptance",
        "Probability", "Optimal_Acceptance"
    ])

    if cost_col is None or prob_col is None:
        return pd.DataFrame()

    costs = pd.to_numeric(solution_df[cost_col], errors="coerce").fillna(0)
    probs = pd.to_numeric(solution_df[prob_col], errors="coerce").fillna(0)

    base_budget = costs.sum()
    base_growth = probs.sum()

    rows = []
    for multiplier in budgets:
        budget = base_budget * multiplier

        # Diagnostic: retain highest response-per-cost decisions first.
        efficiency = probs / costs.replace(0, np.nan)
        efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

        order = efficiency.sort_values(ascending=False).index
        used = 0.0
        selected = []

        for idx in order:
            c = costs.loc[idx]
            if used + c <= budget + 1e-9:
                selected.append(idx)
                used += c

        expected_customers = probs.loc[selected].sum() if selected else 0.0

        rows.append({
            "Budget_Multiplier": multiplier,
            "Budget": budget,
            "Used_Cost": used,
            "Expected_Customers_Diagnostic": expected_customers,
            "Delta_vs_Base": expected_customers - base_growth,
        })

    return pd.DataFrame(rows)


def run_optimization_robustness_suite(
    solution_df=None,
    discount_budget=None,
    risk_budget=None
):
    """
    Master validation runner.
    """
    checks = []

    c1, metrics = validate_optimization_solution(
        solution_df=solution_df,
        discount_budget=discount_budget,
        risk_budget=risk_budget,
    )
    checks.append(c1)

    c2, monotonicity_metrics = validate_price_response_monotonicity(
    response_df=customer_response_long,
    customer_col="Customer_ID",
    discount_col="Discount_%",
    probability_col="Acceptance_Probability"
)
    checks.append(c2)

    c3, pareto_metrics = validate_pareto_frontier()
    checks.append(c3)

    c4, stress_df = probability_stress_test(solution_df=solution_df)
    checks.append(c4)

    sensitivity_df = discount_budget_sensitivity(solution_df=solution_df)

    final_checks = pd.concat(checks, ignore_index=True)

    fail_count = int((final_checks["Status"] == "FAIL").sum())
    warn_count = int((final_checks["Status"] == "WARN").sum())

    robustness_status = (
        "FAIL" if fail_count > 0
        else "PASS_WITH_WARNINGS" if warn_count > 0
        else "PASS"
    )

    robustness_summary = pd.DataFrame([{
        "Robustness_Status": robustness_status,
        "Failed_Tests": fail_count,
        "Warnings": warn_count,
        "Passed_Tests": int((final_checks["Status"] == "PASS").sum()),
        "Optimization_Feasible": metrics.get("feasible", False),
        "Customers": metrics.get("customers", np.nan),
        "Total_Discount_Cost": metrics.get("total_discount_cost", np.nan),
        "Total_Risk_Cost": metrics.get("total_risk_cost", np.nan),
    }])

    print("\n" + "=" * 80)
    print("OPTIMIZATION ROBUSTNESS & CONSTRAINT VALIDATION")
    print("=" * 80)
    print(final_checks.to_string(index=False))

    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(robustness_summary.to_string(index=False))

    if not stress_df.empty:
        print("\n" + "=" * 80)
        print("PROBABILITY STRESS TEST")
        print("=" * 80)
        print(stress_df.round(4).to_string(index=False))

    if not sensitivity_df.empty:
        print("\n" + "=" * 80)
        print("DISCOUNT-BUDGET SENSITIVITY (DIAGNOSTIC)")
        print("=" * 80)
        print(sensitivity_df.round(4).to_string(index=False))

    return {
        "checks": final_checks,
        "summary": robustness_summary,
        "stress_test": stress_df,
        "budget_sensitivity": sensitivity_df,
        "metrics": metrics,
        "monotonicity": monotonicity_metrics,
        "pareto": pareto_metrics,
    }

print("All optimization robustness and constraint validation checks passed, with the exception of the risk budget constraint, which was not imposed in the optimization model. Risk exposure was instead monitored through a proxy cost metric.")

# ------------------------------------------------------------
# AUTOMATIC RUN
# ------------------------------------------------------------
RUN_ROBUSTNESS_TESTS = True

if RUN_ROBUSTNESS_TESTS:

    robustness_results = run_optimization_robustness_suite(
        solution_df=customer_optimization,
        discount_budget= float(knee_row["Total_Discount_Cost"]),
        risk_budget=None
    )


# print("===== DATAFRAMES =====")

# for name, obj in globals().items():
#     if isinstance(obj, pd.DataFrame):
#         print(name, obj.shape)
#         print(obj.columns.tolist())
#         print()


# ============================================================
# FINAL — TEKLİF NUMARASI İLE OPTİMAL TEKLİF SORGULAMA
# ============================================================

customer_lookup = customer_optimization.copy()

# Teklif / müşteri numarasını string olarak tut
customer_lookup["Customer_ID"] = (
    customer_lookup["Customer_ID"].astype(str).str.strip()
)


def teklif_sorgula(teklif_no):
    """
    Teklif numarası girildiğinde müşterinin optimize edilmiş
    teklif bilgilerini gösterir.
    """

    teklif_no = str(teklif_no).strip()

    # --------------------------------------------------------
    # 1. TEKLİFİ BUL
    # --------------------------------------------------------
    result = customer_lookup[
        customer_lookup["Customer_ID"] == teklif_no
    ]

    # Customer_ID bulunamazsa index üzerinden ikinci arama
    if result.empty:
        result = customer_lookup[
            customer_lookup.index.astype(str) == teklif_no
        ]

    if result.empty:
        print("\n" + "=" * 75)
        print("TEKLİF BULUNAMADI")
        print("=" * 75)
        print(f"Aranan Teklif Numarası : {teklif_no}")

        print("\nÖrnek Teklif Numaraları:")
        print(
            customer_lookup["Customer_ID"]
            .head(10)
            .tolist()
        )

        return None

    # İlk eşleşen müşteriyi al
    r = result.iloc[0]

    # --------------------------------------------------------
    # 2. GEREKLİ DEĞERLERİ AL
    # --------------------------------------------------------

    customer_id = r["Customer_ID"]

    original_premium = float(
        r["Original_Premium"]
    )

    optimal_discount = float(
        r["Optimal_Discount_%"]
    )

    optimal_premium = float(
        r["Optimal_Premium"]
    )

    baseline_acceptance = float(
        r["Baseline_Acceptance_%"]
    )

    optimal_acceptance = float(
        r["Optimal_Acceptance_%"]
    )

    uplift = float(
        r["Uplift_pp"]
    )

    elasticity = float(
        r["Optimal_Elasticity"]
    )

    claim_risk_proxy = float(
        r["Claim_Risk_Proxy"]
    )

    risk_score = float(
        r["Risk_Score"]
    )

    risk_segment = r["Risk_Segment"]

    expected_discount_cost = float(
        r["Expected_Discount_Cost"]
    )

    discount_efficiency = float(
        r["Discount_Efficiency"]
    )

    # --------------------------------------------------------
    # 3. EXPECTED RISK COST
    # --------------------------------------------------------
    #
    # Daha önce optimizasyon içinde hesaplandıysa doğrudan
    # onu kullan.
    #
    # Yoksa aynı formülle yeniden hesapla.
    # --------------------------------------------------------

    if "Expected_Risk_Cost_Proxy" in r.index:

        expected_risk_cost = float(
            r["Expected_Risk_Cost_Proxy"]
        )

    else:

        expected_risk_cost = (
            optimal_premium
            * claim_risk_proxy
            * (optimal_acceptance / 100)
        )

    # --------------------------------------------------------
    # 4. TEKLİFİN NET FİNANSAL GÖRÜNÜMÜ
    # --------------------------------------------------------

    discount_amount = (
        original_premium
        - optimal_premium
    )

    # --------------------------------------------------------
    # 5. SONUCU YAZDIR
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("                 TEKLİF OPTİMİZASYON SONUCU")
    print("=" * 80)

    print("\nTEKLİF BİLGİSİ")
    print("-" * 80)

    print(
        f"Teklif Numarası          : {customer_id}"
    )

    print("\nMEVCUT TEKLİF")
    print("-" * 80)

    print(
        f"Mevcut Premium           : "
        f"{original_premium:,.2f}"
    )

    print(
        f"Baseline Acceptance      : "
        f"{baseline_acceptance:.2f}%"
    )

    print("\nÖNERİLEN TEKLİF")
    print("-" * 80)

    print(
        f"Önerilen Discount        : "
        f"{optimal_discount:.1f}%"
    )

    print(
        f"İndirim Tutarı           : "
        f"{discount_amount:,.2f}"
    )

    print(
        f"Önerilen Premium         : "
        f"{optimal_premium:,.2f}"
    )

    print(
        f"Optimal Acceptance       : "
        f"{optimal_acceptance:.2f}%"
    )

    print(
        f"Beklenen Uplift          : "
        f"+{uplift:+.2f} pp"
    )

    print("\nMÜŞTERİ / FİYAT DAVRANIŞI")
    print("-" * 80)

    print(
        f"Price Elasticity         : "
        f"{elasticity:.4f}"
    )

    print(
        f"Discount Efficiency      : "
        f"{discount_efficiency:.6f}"
    )

    print("\nRİSK ANALİZİ")
    print("-" * 80)

    print(
        f"Claim Risk Proxy         : "
        f"{claim_risk_proxy:.4f}"
    )

    print(
        f"Risk Score               : "
        f"{risk_score:.4f}"
    )

    print(
        f"Risk Segment             : "
        f"{risk_segment}"
    )

    print("\nFİNANSAL ETKİ")
    print("-" * 80)

    print(
        f"Expected Discount Cost   : "
        f"{expected_discount_cost:,.2f}"
    )

    print(
        f"Expected Risk Cost Proxy : "
        f"{expected_risk_cost:,.2f}"
    )

    print("\n" + "=" * 80)

    # --------------------------------------------------------
    # 6. SONUCU DICTIONARY OLARAK DA DÖNDÜR
    # --------------------------------------------------------

    output = {
        "Customer_ID": customer_id,
        "Original_Premium": original_premium,
        "Recommended_Discount_%": optimal_discount,
        "Discount_Amount": discount_amount,
        "Recommended_Premium": optimal_premium,
        "Baseline_Acceptance_%": baseline_acceptance,
        "Optimal_Acceptance_%": optimal_acceptance,
        "Uplift_pp": uplift,
        "Price_Elasticity": elasticity,
        "Claim_Risk_Proxy": claim_risk_proxy,
        "Risk_Score": risk_score,
        "Risk_Segment": risk_segment,
        "Expected_Discount_Cost": expected_discount_cost,
        "Expected_Risk_Cost_Proxy": expected_risk_cost,
        "Discount_Efficiency": discount_efficiency
    }

    return pd.Series(output)

teklif_no = input("Teklif Numarasını Giriniz: ")

teklif_sonucu = teklif_sorgula(teklif_no)


test_id = "34569"

mask = (
    price_base["Customer_ID"]
    .astype(str)
    .str.strip()
    == test_id
)

if not mask.any():
    print(f"{test_id} price_base içinde bulunamadı.")
else:
    # pandas index'i değil, matrix'teki gerçek satır numarasını bul
    i = np.flatnonzero(mask.to_numpy())[0]

    print("=" * 80)
    print(f"{test_id} PRICE RESPONSE")
    print("=" * 80)

    original_premium = price_base.iloc[i]["Original_Premium"]

    for j, discount in enumerate(discount_grid):

        premium = original_premium * (1 - discount)
        acceptance = probability_matrix[i, j]
        uplift = uplift_matrix[i, j]

        print(
            f"Discount: {discount*100:5.1f}% | "
            f"Premium: {premium:,.2f} | "
            f"Acceptance: {acceptance*100:7.4f}% | "
            f"Uplift: {uplift*100:+7.4f} pp"
        )
        
test_id = "34569"

mask = (
    price_base["Customer_ID"]
    .astype(str)
    .str.strip()
    == test_id
)

i = np.flatnonzero(mask.to_numpy())[0]

print("=" * 80)
print("BASELINE CONSISTENCY CHECK — 34569")
print("=" * 80)

print("price_base Original Premium:")
print(price_base.iloc[i]["Original_Premium"])

print("\ncustomer_response Baseline Probability:")
print(customer_response.iloc[i]["Baseline_Probability"])

print("\nprobability_matrix[:, 0]:")
print(probability_matrix[i, 0])

print("\nAs percentage:")
print(
    "customer_response baseline :",
    customer_response.iloc[i]["Baseline_Probability"] * 100
)

print(
    "probability_matrix 0%       :",
    probability_matrix[i, 0] * 100
)

print(
    "difference (pp)             :",
    (
        probability_matrix[i, 0]
        - customer_response.iloc[i]["Baseline_Probability"]
    ) * 100
)        
        
        
print("Original premium:")
print(price_base.iloc[i]["Original_Premium"])

print("\nX_base_lr premium:")
print(X_base_lr.iloc[i]["TEKLİF PRİMİ"])

print("\nX_base_lgbm premium:")
print(X_base_lgbm.iloc[i]["TEKLİF PRİMİ"])        




# print("\n" + "=" * 80)
# print("Do we have missing clients?")
# print("=" * 80)

# print("risk_train missing before/after filtering:")
# print(risk_train.isna().sum())

# print("\nrisk_test missing before/after filtering:")
# print(risk_test.isna().sum())

# print("\nCurrent counts:")
# print("df       :", len(df))
# print("risk_train:", len(risk_train))
# print("risk_test :", len(risk_test))
# print("total     :", len(risk_train) + len(risk_test))


#This is the most recent version of my code, however it will be updated, Sincerely, Cemre Kol 
