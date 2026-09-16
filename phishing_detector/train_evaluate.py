import logging
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imbalanced-learn chua cai. Bo qua SMOTE (pip install imbalanced-learn).")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost khong kha dung.")

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Nguong mat can bang: neu minority/majority < nguong nay thi apply SMOTE
IMBALANCE_THRESHOLD = 0.6     # ty le < 60% la mat can bang
SVM_SUBSAMPLE_LIMIT = 150_000  # Tang len de tan dung nhieu mau hon voi 500K
SMOTE_LARGE_THRESHOLD = 100_000  # Neu train set > nguong nay thi dung RandomOverSampler thay SMOTE


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

def get_models(class_ratio: float = 1.0) -> Dict[str, object]:
    """
    Khởi tạo các mô hình ML.

    Args:
        class_ratio: ty le benign/phishing (dung cho scale_pos_weight cua XGBoost).
                     VD: 392924 benign / 156422 phishing = 2.51
    """
    models = {
        "Logistic Regression": Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                random_state=42,
                class_weight="balanced",
                C=0.5,               # tang regularization de giam overfit
                solver="saga",       # nhanh hon voi big data, ho tro n_jobs
                n_jobs=-1,           # su dung toan bo CPU core
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,          # tang tu 12 - du lieu lon cho phep tree sau hon
            min_samples_split=20,  # tang tu 10 - can nhieu mau hon moi split
            min_samples_leaf=10,   # tang tu 5  - la cay can du mau tranh overfit
            random_state=42,
            class_weight="balanced",
        ),
        # LinearSVC nhanh hon RBF, scale tot hon o dataset lon
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    max_iter=3000,
                    random_state=42,
                    class_weight="balanced",
                )
            )),
        ]),
    }

    if XGBOOST_AVAILABLE:
        # scale_pos_weight = so luong benign / so luong phishing
        # Giup XGBoost chu y hon toi lop phishing (thieu so)
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,          # tang tu 300 - hoc sau hon voi big data
            max_depth=8,               # tang tu 7
            learning_rate=0.05,        # giam tu 0.08 - hoc can than hon
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=5,        # tang tu 3 - chong overfit voi big data
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=1.5,            # tang L2 tu 1.0
            tree_method="hist",        # nhanh gap 3-5x voi dataset lon
            scale_pos_weight=class_ratio,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return models


# ─────────────────────────────────────────────
# Oversampling
# ─────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ap dung SMOTE de can bang tap huan luyen.
    Chi ap dung neu ty le minority < IMBALANCE_THRESHOLD.
    """
    counts = np.bincount(y_train.astype(int))
    if len(counts) < 2:
        return X_train, y_train

    minority = counts.min()
    majority = counts.max()
    ratio = minority / majority

    if ratio >= IMBALANCE_THRESHOLD:
        logger.info(f"  [SMOTE] Du lieu da can bang ({ratio:.1%}), bo qua SMOTE.")
        return X_train, y_train

    if not SMOTE_AVAILABLE:
        logger.warning("  [SMOTE] imbalanced-learn chua cai, bo qua.")
        return X_train, y_train

    logger.info(
        f"  [SMOTE] Mat can bang phat hien ({ratio:.1%}). "
        f"Dang oversampling lop thieu so ({minority} -> {majority} mau)..."
    )
    try:
        # k_neighbors <= so mau thieu so - 1
        k = min(5, minority - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        new_counts = np.bincount(y_res.astype(int))
        logger.info(
            f"  [SMOTE] Sau SMOTE: benign={new_counts[0]:,}, phishing={new_counts[1]:,}"
        )
        return X_res, y_res
    except Exception as e:
        logger.warning(f"  [SMOTE] Loi khi SMOTE: {e}. Dung du lieu goc.")
        return X_train, y_train


def apply_oversampling(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chon phuong phap can bang du lieu phu hop voi kich thuoc tap huan luyen:
      - Dataset lon (> SMOTE_LARGE_THRESHOLD): dung RandomOverSampler (nhanh ~10x)
      - Dataset nho                          : dung SMOTE (mau tong hop da dang hon)
    """
    counts = np.bincount(y_train.astype(int))
    if len(counts) < 2:
        return X_train, y_train

    minority = counts.min()
    majority = counts.max()
    ratio = minority / majority

    if ratio >= IMBALANCE_THRESHOLD:
        logger.info(f"  [OVERSAMPLE] Du lieu da can bang ({ratio:.1%}), bo qua oversampling.")
        return X_train, y_train

    if len(X_train) > SMOTE_LARGE_THRESHOLD:
        # Dataset lon: RandomOverSampler nhanh hon nhieu, khong can tinh kNN
        try:
            from imblearn.over_sampling import RandomOverSampler
            logger.info(
                f"  [OVERSAMPLE] Dataset lon ({len(X_train):,} mau). "
                f"Dung RandomOverSampler ({minority:,} -> {majority:,} mau)..."
            )
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            new_counts = np.bincount(y_res.astype(int))
            logger.info(
                f"  [OVERSAMPLE] Sau ROS: benign={new_counts[0]:,}, phishing={new_counts[1]:,}"
            )
            return X_res, y_res
        except ImportError:
            logger.warning("  [OVERSAMPLE] imbalanced-learn chua cai. Thu SMOTE...")
            return apply_smote(X_train, y_train)
        except Exception as e:
            logger.warning(f"  [OVERSAMPLE] Loi RandomOverSampler: {e}. Thu SMOTE...")
            return apply_smote(X_train, y_train)
    else:
        return apply_smote(X_train, y_train)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
    feature_set: str = "",
    use_smote: bool = True,
) -> dict:
    """Huấn luyện một mô hình và trả về các chỉ số đánh giá."""

    X_tr, y_tr = X_train, y_train

    # Apply oversampling (chi tren tap train, khong cham den tap test)
    if use_smote:
        X_tr, y_tr = apply_oversampling(X_tr, y_tr)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)

    result = {
        "Model": model_name,
        "Feature Set": feature_set,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    return result, model, y_pred


def build_feature_combinations(
    df_features: pd.DataFrame,
    lexical_cols: List[str],
    host_cols: List[str],
    content_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Tạo các tổ hợp đặc trưng chuẩn theo bài báo:
      - X1 (Lexical): 26 đặc trưng URL string
      - X1+X2: Lexical + Host-based (WHOIS)
      - X1+X3: Lexical + Content-based (HTTP)
      - X1+X2+X3: Toàn bộ 29 đặc trưng
    """
    L = [c for c in lexical_cols  if c in df_features.columns]
    H = [c for c in host_cols     if c in df_features.columns]
    C = [c for c in content_cols  if c in df_features.columns]

    combos = {}
    if L:
        combos["X1 (Lexical)"] = df_features[L]
    if L and H:
        combos["X1+X2"] = df_features[L + H]
    if L and C:
        combos["X1+X3"] = df_features[L + C]
    if L and H and C:
        combos["X1+X2+X3"] = df_features[L + H + C]

    filtered = {name: df for name, df in combos.items() if df.shape[1] > 0}
    return filtered


def run_training_pipeline(
    df_features: pd.DataFrame,
    y: pd.Series,
    lexical_cols: List[str],
    host_cols: List[str],
    content_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Chạy pipeline huấn luyện và đánh giá cho tất cả mô hình × tổ hợp.
    """
    counts = y.value_counts()
    n_benign   = int(counts.get(0, 0))
    n_phishing = int(counts.get(1, 1))
    class_ratio = n_benign / max(n_phishing, 1)

    logger.info(f"  [RATIO] Benign={n_benign:,} | Phishing={n_phishing:,} | "
                f"Ratio={class_ratio:.2f} -> scale_pos_weight={class_ratio:.2f}")

    combos = build_feature_combinations(
        df_features, lexical_cols, host_cols, content_cols
    )
    if not combos:
        raise ValueError("Khong co to hop dac trung nao hop le!")

    models = get_models(class_ratio=class_ratio)
    all_results = []
    trained_models = {}

    total = len(models) * len(combos)
    count = 0

    for feat_name, X in combos.items():
        X_arr = X.fillna(-1).values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y.values, test_size=test_size,
            random_state=random_state, stratify=y
        )

        for model_name, model in models.items():
            count += 1
            logger.info(f"[{count}/{total}] Đang huấn luyện: {model_name} | {feat_name}")

            try:
                from sklearn.base import clone
                m = clone(model)

                # Subsample SVM tren tap train neu qua lon
                X_tr_m, y_tr_m = X_train, y_train
                if model_name == "SVM" and len(X_train) > SVM_SUBSAMPLE_LIMIT:
                    idx = np.random.choice(len(X_train), SVM_SUBSAMPLE_LIMIT, replace=False)
                    X_tr_m, y_tr_m = X_train[idx], y_train[idx]
                    logger.info(f"  [SVM] Subsample {SVM_SUBSAMPLE_LIMIT:,} mau de train nhanh.")

                result, trained_m, y_pred = evaluate_model(
                    m, X_tr_m, X_test, y_tr_m, y_test,
                    model_name=model_name,
                    feature_set=feat_name,
                    use_smote=use_smote,
                )
                all_results.append(result)
                key = f"{model_name} | {feat_name}"
                trained_models[key] = {
                    "model": trained_m,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "feature_cols": list(X.columns),
                }
                logger.info(
                    f"  -> Acc={result['Accuracy']:.4f} | "
                    f"P={result['Precision']:.4f} | "
                    f"R={result['Recall']:.4f} | "
                    f"F1={result['F1-Score']:.4f}"
                )
            except Exception as e:
                logger.error(f"Loi khi huan luyen {model_name} tren {feat_name}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # Renaming columns for display
    results_df = results_df.rename(columns={"Feature Set": "Feature Set"})

    # Tim mo hinh tot nhat
    best_row = results_df.iloc[0]
    best_key = f"{best_row['Model']} | {best_row['Feature Set']}"
    best_model = {
        "name": best_row["Model"],
        "feature_set": best_row["Feature Set"],
        "metrics": best_row.to_dict(),
        "model": trained_models.get(best_key, {}).get("model"),
        "feature_cols": trained_models.get(best_key, {}).get("feature_cols", []),
    }

    return results_df, best_model, trained_models


if __name__ == "__main__":
    # Quick test voi du lieu gia mat can bang
    np.random.seed(42)
    n = 500
    X_fake = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    # Mat can bang 20/80
    y_fake = pd.Series(np.random.choice([0, 1], size=n, p=[0.8, 0.2]))

    results, best, _ = run_training_pipeline(
        X_fake, y_fake,
        [f"f{i}" for i in range(5)], [], [],
        use_smote=True,
    )
    print(results.to_string())
    print(f"\nBest model: {best['name']} | {best['feature_set']}")
