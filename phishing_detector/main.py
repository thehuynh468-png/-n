import argparse
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Set stdout/stderr to UTF-8 to support Vietnamese + special chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Local modules
from feature_extraction import (
    extract_features,
    LEXICAL_COLS, HOST_COLS, CONTENT_COLS,
)
from train_evaluate import run_training_pipeline
from report_generator import generate_report

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEP1 = "=" * 60
SEP2 = "-" * 60


# ─────────────────────────────────────────────
# Step 1: Load data
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Doc file CSV, kiem tra va lam sach du lieu."""
    logger.info(f"[I/O] Doc file: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"  Tong so hang goc: {len(df)}")
    logger.info(f"  Cac cot: {list(df.columns)}")

    # Chuan hoa ten cot
    df.columns = [c.strip().lower() for c in df.columns]

    # Kiem tra cot bat buoc
    if "url" not in df.columns:
        raise ValueError("File CSV phai co cot 'url'.")

    # Ho tro cot 'type' (vi du: malicious_phish.csv) bang cach doi ten thanh 'label'
    if "label" not in df.columns and "type" in df.columns:
        logger.info("  [MAP] Phat hien cot 'type', doi ten thanh 'label'.")
        df = df.rename(columns={"type": "label"})

    if "label" not in df.columns:
        raise ValueError("File CSV phai co cot 'label' hoac 'type'.")

    # Xu ly gia tri thieu
    missing_url   = df["url"].isna().sum()
    missing_label = df["label"].isna().sum()

    if missing_url > 0:
        logger.warning(f"  [WARN] Bo {missing_url} hang thieu URL.")
        df = df.dropna(subset=["url"])

    if missing_label > 0:
        logger.warning(f"  [WARN] Bo {missing_label} hang thieu label.")
        df = df.dropna(subset=["label"])

    df["url"] = df["url"].astype(str).str.strip()

    # Chuyen doi label: ho tro ca so (0/1) lan chu (bad/good/phishing/benign/defacement/malware)
    label_lower = df["label"].astype(str).str.strip().str.lower()
    if label_lower.isin(["0", "1"]).all():
        df["label"] = label_lower.astype(int)
    else:
        str_map = {
            # Malicious / Phishing
            "bad": 1, "phishing": 1, "malicious": 1, "spam": 1, "1": 1,
            # Cac loai tan cong khac cung la malicious
            "defacement": 1, "malware": 1, "ransomware": 1, "exploit": 1,
            "scam": 1, "fraud": 1, "attack": 1,
            # Benign / Safe
            "good": 0, "benign": 0, "legitimate": 0, "safe": 0, "0": 0,
            "clean": 0, "whitelist": 0,
        }
        mapped = label_lower.map(str_map)
        unknown = mapped.isna().sum()
        if unknown > 0:
            unknown_vals = label_lower[mapped.isna()].unique().tolist()
            logger.warning(f"  [WARN] {unknown} label khong nhan ra: {unknown_vals}. Bo qua.")
            df = df[mapped.notna()].copy()
            mapped = mapped.dropna()
        df["label"] = mapped.astype(int)
        logger.info("  [MAP] phishing/defacement/malware->1 | benign->0")

    # Loc label hop le
    valid_labels = {0, 1}
    invalid_mask = ~df["label"].isin(valid_labels)
    if invalid_mask.sum() > 0:
        logger.warning(f"  [WARN] Bo {invalid_mask.sum()} hang label khong hop le.")
        df = df[~invalid_mask]

    df = df.reset_index(drop=True)

    phishing = (df["label"] == 1).sum()
    benign   = (df["label"] == 0).sum()
    logger.info(f"  [OK] Sau lam sach: {len(df)} URL")
    logger.info(f"    Phishing (label=1): {phishing} ({phishing/len(df)*100:.1f}%)")
    logger.info(f"    Benign   (label=0): {benign}   ({benign/len(df)*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# Step 2: Feature extraction
# ─────────────────────────────────────────────

def extract_all_features(
    df: pd.DataFrame,
    fetch_whois: bool = False,
    fetch_page: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Trich xuat dac trung song song cho toan bo dataset."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    extra = ""
    if fetch_whois: extra += " [WHOIS=ON]"
    if fetch_page:  extra += " [HTTP=ON]"
    logger.info(f"[FEAT] Trich xuat dac trung ({len(df)} URL){extra}...")

    urls = df["url"].tolist()
    results = [None] * len(urls)

    def _extract(idx_url):
        idx, url = idx_url
        try:
            return idx, extract_features(url, fetch_whois=fetch_whois, fetch_page=fetch_page)
        except Exception as e:
            logger.debug(f"Loi trich xuat URL {url}: {e}")
            return idx, {}

    workers = 1 if (fetch_whois or fetch_page) else max_workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract, (i, u)): i for i, u in enumerate(urls)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            idx, feat = future.result()
            results[idx] = feat

    features_df = pd.DataFrame(results)
    logger.info(f"  [OK] Dac trung da trich xuat: {features_df.shape[1]} cot")
    return features_df


# ─────────────────────────────────────────────
# Step 4: Statistical analysis
# ─────────────────────────────────────────────

def analyze_features(features_df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """In thong ke mo ta va so sanh phishing vs benign."""
    lexical_cols_present = [c for c in LEXICAL_COLS if c in features_df.columns]
    df_lex = features_df[lexical_cols_present]

    logger.info("\n" + SEP1)
    logger.info("[STATS] THONG KE MO TA - DAC TRUNG LEXICAL")
    logger.info(SEP1)

    stats = df_lex.describe().T[["mean", "std", "min", "max"]]
    print("\n" + stats.to_string())

    # So sanh phishing vs benign
    df_combined = df_lex.copy()
    df_combined["label"] = y.values

    logger.info("\n" + SEP2)
    logger.info("[COMPARE] SO SANH TRUNG BINH: Phishing vs Benign")
    logger.info(SEP2)

    comparison = df_combined.groupby("label")[lexical_cols_present].mean().T
    comparison.columns = ["Benign (0)", "Phishing (1)"]
    comparison["Delta (Phishing-Benign)"] = comparison["Phishing (1)"] - comparison["Benign (0)"]
    print("\n" + comparison.round(3).to_string())

    logger.info("\n" + SEP2)
    logger.info("[VN] NHAN XET PHISHING TAI VIET NAM")
    logger.info(SEP2)
    vietnam_obs = """
    1. TEN MIEN GIA MAO:
       URL phishing Viet Nam hiem khi dung .vn (ton kem, can xac thuc).
       Thay vao do dung .com/.net/.top/.xyz voi subdomain gia ten ngan hang:
         vietcombank.secure-login.com, techcombank-auth.net, bidv.mobile-banking.top

    2. TU KHOA TIENG VIET:
       dangnhap, thanhtoan, baomat, taikhoan, xacnhan, matkhau
       He thong da nhan dien qua dac trung 'has_sensitive_keyword'.

    3. DO DAI URL:
       URL phishing Viet Nam trung binh dai 80-150 ky tu
       (quoc te: ~75 ky tu). Them nhieu query params gia (token, session, ref).

    4. DICH VU RUT GON:
       Ke tan cong dung bit.ly, tinyurl qua SMS/Zalo/Facebook
       de che giau URL that, dac biet pho bien trong lua dao mobile banking.
    """
    print(vietnam_obs)

    return stats


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phishing URL Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=str, default="phishing_urls.csv",
        help="Duong dan file CSV (mac dinh: phishing_urls.csv)"
    )
    parser.add_argument(
        "--output", type=str, default="BaoCao_Phishing_ML.docx",
        help="Duong dan file bao cao .docx dau ra"
    )
    parser.add_argument(
        "--whois", action="store_true",
        help="Bat tra cuu WHOIS (cham)"
    )
    parser.add_argument(
        "--fetch-page", action="store_true",
        help="Bat tai trang web de trich xuat dac trung content"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="So luong song song khi trich xuat lexical (mac dinh: 8)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Chi dung N mau dau tien (de test nhanh)"
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info("[START] BAT DAU PHISHING DETECTION PIPELINE")
    logger.info(SEP1)

    # ── Step 1: Load data ──
    df = load_data(args.csv)

    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"  [SAMPLE] Chi dung {len(df)} mau (--sample {args.sample})")

    y = df["label"]

    # ── Step 2: Feature extraction ──
    features_df = extract_all_features(
        df,
        fetch_whois=args.whois,
        fetch_page=getattr(args, "fetch_page", False),
        max_workers=args.workers,
    )

    # ── Step 3: Build feature DataFrames ──
    logger.info("\n" + SEP2)
    logger.info("[BUILD] Chuan bi to hop dac trung...")

    actual_lex     = [c for c in LEXICAL_COLS  if c in features_df.columns]
    actual_host    = [c for c in HOST_COLS      if c in features_df.columns]
    actual_content = [c for c in CONTENT_COLS   if c in features_df.columns]

    logger.info(f"  Lexical (X1) : {len(actual_lex)} dac trung")
    logger.info(f"  Host    (X2) : {len(actual_host)} dac trung")
    logger.info(f"  Content (X3) : {len(actual_content)} dac trung")

    # ── Step 4: Analysis ──
    stats_df = analyze_features(features_df, y)

    # ── Step 5: Training ──
    logger.info("\n" + SEP1)
    logger.info("[TRAIN] HUAN LUYEN VA SO SANH CAC MO HINH ML")
    logger.info(SEP1)

    results_df, best_model, trained_models = run_training_pipeline(
        features_df, y,
        lexical_cols=actual_lex,
        host_cols=actual_host,
        content_cols=actual_content,
    )

    # ── Print comparison table ──
    logger.info("\n" + SEP1)
    logger.info("[RESULT] BANG SO SANH KET QUA")
    logger.info(SEP1)
    print("\n" + results_df.to_string(index=False))

    # ── Step 6: Best model ──
    logger.info("\n" + SEP2)
    logger.info("[BEST] MO HINH TOT NHAT")
    logger.info(SEP2)
    m = best_model["metrics"]
    logger.info(f"  Mo hinh       : {best_model['name']}")
    logger.info(f"  To hop dac trung: {best_model['feature_set']}")
    logger.info(f"  Accuracy      : {m.get('Accuracy',  0):.4f}")
    logger.info(f"  Precision     : {m.get('Precision', 0):.4f}")
    logger.info(f"  Recall        : {m.get('Recall',    0):.4f}")
    logger.info(f"  F1-Score      : {m.get('F1-Score',  0):.4f}")

    # ── Step 7: Generate report ──
    logger.info("\n" + SEP2)
    logger.info(f"[REPORT] Dang tao bao cao Word: {args.output}")

    # Thu luu vao ten goc truoc
    output_path = args.output
    try:
        generate_report(
            output_path=output_path,
            results_df=results_df,
            best_model=best_model,
            stats_df=stats_df,
            total_urls=len(df),
            phishing_count=int((y == 1).sum()),
            benign_count=int((y == 0).sum()),
            feature_cols=actual_lex,
            dataset_path=args.csv,
        )
    except PermissionError:
        # File dang mo trong Word -> luu vao ten co timestamp
        from datetime import datetime
        ts = datetime.now().strftime("%H%M%S")
        stem = Path(args.output).stem
        output_path = f"{stem}_{ts}.docx"
        logger.warning(
            f"  [WARN] '{args.output}' dang bi khoa (Word dang mo?)."
            f" Luu vao '{output_path}' thay the."
        )
        generate_report(
            output_path=output_path,
            results_df=results_df,
            best_model=best_model,
            stats_df=stats_df,
            total_urls=len(df),
            phishing_count=int((y == 1).sum()),
            benign_count=int((y == 0).sum()),
            feature_cols=actual_lex,
            dataset_path=args.csv,
        )

    elapsed = time.time() - start_time
    logger.info(f"\n[DONE] HOAN THANH! Thoi gian: {elapsed:.1f}s")
    logger.info(f"  Bao cao: {output_path}")
    logger.info(f"  Log file: pipeline.log")


if __name__ == "__main__":
    main()
