"""
api.py — FastAPI backend cho PhishGuard DL models
Chạy: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).parent
MODEL_NAMES = ["CNN_BiLSTM", "CharCNN", "BiLSTM"]

_models: dict = {}
_tokenizer: dict | None = None


# ──Khởi tạo cấu hình và Nạp các mô hình Học sâu khi API khởi động ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tất cả models khi khởi động."""
    global _tokenizer
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        tok_path = BASE / "tokenizer.json"
        if tok_path.exists():
            with open(tok_path, encoding="utf-8") as f:
                _tokenizer = json.load(f)
            print(f"[API] Tokenizer loaded (vocab={_tokenizer['vocab_size']}, max_len={_tokenizer['max_len']})")
        else:
            print("[API] WARNING: tokenizer.json not found")

        for name in MODEL_NAMES:
            path = BASE / f"{name}_phishing.keras"
            if path.exists():
                _models[name] = tf.keras.models.load_model(str(path))
                print(f"[API] Model loaded: {name}")
            else:
                print(f"[API] WARNING: {name}_phishing.keras not found")

    except ImportError:
        print("[API] ERROR: TensorFlow not installed")

    print(f"[API] Ready — models: {list(_models.keys())}")
    yield
    _models.clear()


app = FastAPI(
    title="PhishGuard API",
    description="Deep Learning backend for phishing URL detection",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    url: str
    model_name: str = "CNN_BiLSTM"
    threshold: float = 0.28   


class PredictResponse(BaseModel):
    url: str
    normalized_url: str
    probability: float
    is_phishing: bool
    label: str
    confidence_pct: float
    model: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    tokenizer_ready: bool


# ── Helpers ───────────────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    """Strip schema vì training data không có http:// cho benign URLs."""
    url = url.strip()
    for schema in ["https://", "http://", "ftp://", "ftps://"]:
        if url.lower().startswith(schema):
            url = url[len(schema):]
            break
    return url


# Domain gốc thực sự uy tín (chỉ root domain, không phải subdomain hosting)
_SAFE_ROOTS = {
    "google.com", "google.com.vn",
    "youtube.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com",
    "github.com", "microsoft.com", "apple.com",
    "amazon.com", "wikipedia.org", "stackoverflow.com",
    "netflix.com", "spotify.com", "reddit.com",
    "tiktok.com", "zalo.me",
    "paypal.com",
    "vietcombank.com.vn", "techcombank.com.vn",
    "bidv.com.vn", "mbbank.com.vn",
    "vpbank.com.vn", "agribank.com.vn",
    "vnpay.vn", "momo.vn",
    "nps.gov", "rsc.org",
}

# Hosting platform bị hacker lạm dụng để host phishing
# → KHÔNG whitelist dù domain gốc uy tín
_HOSTING_PLATFORMS = {
    # Google services bị lạm dụng
    "sites.google.com",
    "docs.google.com",
    "drive.google.com",
    "drawings.google.com",
    "forms.google.com",
    "storage.googleapis.com",
    "translate.google.com",
    "forms.gle",
    "web.archive.org",
    # Microsoft / Adobe
    "new.express.adobe.com",
    "express.adobe.com",
    "my.sharepoint.com",
    "1drv.ms", "onedrive.live.com",
    # Firebase / Google Cloud
    "firebaseapp.com", "web.app",
    "firebasestorage.googleapis.com",
    "appspot.com",
    # Hosting tĩnh phổ biến
    "glitch.me", "repl.co", "replit.app",
    "netlify.app", "vercel.app", "pages.dev",
    "framer.app", "framer.ai", "framer.website",
    "webflow.io", "carrd.co", "strikingly.com",
    "github.io", "gitlab.io",
    # Website builder bị lạm dụng
    "wixstudio.com", "wix.com", "wixsite.com",
    "weebly.com", "webs.com",
    "squarespace.com", "wordpress.com",
    "blogspot.com", "tumblr.com",
    # Công cụ khác
    "typeform.com", "jotform.com",
    "notion.so", "notion.site",
    "airtable.com",
    "linktree.ee", "linktr.ee",
    "surveymonkey.com",
}

# Subdomain uy tín thực sự (mail, www, support...)
_SAFE_PREFIXES = {"www.", "mail.", "support.", "help.", "m.", "en."}


def _is_safe_domain(domain: str) -> bool:
    """
    Kiểm tra xem domain có thực sự uy tín không.
    - Phải là root domain hoặc subdomain uy tín (www/mail/m...)
    - KHÔNG phải hosting platform bị lạm dụng
    """
    d = domain.lower().strip()

    # Bỏ qua nếu là hosting platform
    if d in _HOSTING_PLATFORMS:
        return False
    for hp in _HOSTING_PLATFORMS:
        if d.endswith("." + hp):
            return False

    # Root domain chính xác
    if d in _SAFE_ROOTS:
        return True

    # Subdomain uy tín (www.google.com, mail.google.com, m.facebook.com...)
    for prefix in _SAFE_PREFIXES:
        if d.startswith(prefix):
            root = d[len(prefix):]
            if root in _SAFE_ROOTS:
                return True

    return False


def heuristic_score(url: str) -> float:
    """
    Rule-based phishing score [0..1].
    Phân biệt keyword trong domain vs path để giảm false positive.
    """
    import re
    u = url.lower().strip()

    # Tách domain và path
    domain = u.split("/")[0].split("?")[0].split("#")[0]
    path = u[len(domain):]

    # ── Kiểm tra domain uy tín ────────────────────────────────────
    if _is_safe_domain(domain):
        # Chỉ cộng điểm nếu path có dấu hiệu THỰC SỰ nguy hiểm
        domain_spoof = any(brand in domain for brand in
            ["paypal", "facebook", "apple", "microsoft", "vietcombank", "momo"])
        if not domain_spoof:
            return 0.05

    score = 0.0

    # ── URL Shortener ─────────────────────────────────────────────
    # Không thể biết đích đến thực → tăng nghi ngờ
    shorteners = [
        "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly",
        "l.ead.me", "q-r.to", "qr.ae", "rebrand.ly", "rb.gy",
        "is.gd", "buff.ly", "dlvr.it", "ff.im", "su.pr",
        "short.link", "cutt.ly", "tiny.cc", "tr.im",
        # Shortener phổ biến ở châu Á / VN
        "vo.la", "t.ly", "s.id", "lc.chat",
        "zws.im", "x.co", "u.to", "clicky.me",
        "budurl.com", "ping.fm", "post.ly",
    ]
    for s in shorteners:
        if domain == s or domain.endswith("." + s):
            score += 0.40
            break

    # ── IP address thay domain ────────────────────────────────────
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", domain):
        score += 0.50

    # IP dạng xxx-xxx-xxx-xxx.cprapid.com
    if re.search(r"\d+-\d+-\d+-\d+\.", domain):
        score += 0.45

    # ── Hosting platform bị lạm dụng ─────────────────────────────
    hosting_hit = False
    for hp in _HOSTING_PLATFORMS:
        if domain == hp or domain.endswith("." + hp):
            score += 0.35
            hosting_hit = True
            break

    # ── Platform-abuse pattern (phishing hiện đại) ────────────────
    # Phishing dùng hosting uy tín nhưng path rất dài/ngẫu nhiên
    if hosting_hit:
        # Path dài bất thường (thường chứa ID ngẫu nhiên)
        if len(path) > 30:
            score += 0.15
        # Google Docs/Drawings với ID dài
        if "docs.google.com" in domain or "drawings.google.com" in domain or "drive.google.com" in domain:
            if re.search(r"/d/[A-Za-z0-9_-]{15,}", path):
                score += 0.20
        # Firebase với subdomain số/random
        if domain.endswith(".web.app") or domain.endswith(".firebaseapp.com"):
            subdomain = domain.split(".")[0]
            if re.search(r"\d{6,}", subdomain) or len(subdomain) > 15:
                score += 0.15
        # Weebly/Wix với subdomain bao gồm brand hoặc keyword
        if domain.endswith(".weebly.com") or domain.endswith(".wixsite.com"):
            subdomain = domain.split(".")[0]
            if any(kw in subdomain for kw in ["bank", "secure", "login", "verify",
                                               "account", "support", "update",
                                               "gov", "official", "vietcombank",
                                               "momo", "zalo", "paypal"]):
                score += 0.25

    # ── Keyword trong DOMAIN → rất đáng ngờ ──────────────────────
    domain_risky_kw = [
        "login", "verify", "secure", "account", "update", "confirm",
        "banking", "paypal", "password", "signin", "wallet",
        "dangnhap", "baomat", "taikhoan", "xacnhan",
        "activation", "authenticate", "credential", "reset-password",
        "user-verification", "client-login",
    ]
    d_hits = sum(1 for kw in domain_risky_kw if kw in domain)
    score += min(d_hits * 0.20, 0.50)

    # ── Keyword trong PATH ────────────────────────────────────────
    path_risky_kw = [
        "login", "verify", "secure-account", "update-account",
        "webscr", "cgi-bin", "wp-admin", "cmd=_login",
        "password", "signin", "authenticate", "credential",
        "banking", "account-suspended", "confirm-identity",
    ]
    p_hits = sum(1 for kw in path_risky_kw if kw in path)
    score += min(p_hits * 0.08, 0.24)

    # ── TLD đáng ngờ ─────────────────────────────────────────────
    suspicious_tlds = [
        ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top",
        ".click", ".link", ".site", ".online", ".work", ".loan",
        ".club", ".biz", ".cc", ".lat", ".ru", ".cn", ".ws",
        ".icu", ".vip", ".zip", ".mov", ".fit", ".gdn",
        ".stream", ".download", ".racing", ".trade",
    ]
    if any(domain.endswith(t) for t in suspicious_tlds):
        score += 0.25

    # ── Nhiều dấu gạch ngang trong domain ────────────────────────
    if domain.count("-") >= 2: score += 0.10
    if domain.count("-") >= 4: score += 0.15

    # ── Nhiều subdomain ───────────────────────────────────────────
    dots = domain.count(".")
    if dots >= 3: score += 0.10
    if dots >= 5: score += 0.15

    # ── Ký tự @ ──────────────────────────────────────────────────
    if "@" in u:
        score += 0.35

    # ── Random hex/base64 trong path (tracking/redirect token) ───
    if re.search(r"/[0-9a-f]{16,}", path):
        score += 0.12
    if re.search(r"/[0-9a-zA-Z]{28,}", path):
        score += 0.10
    # Fragment dài (thường dùng trong phishing redirect)
    if "#" in u:
        fragment = u.split("#", 1)[1]
        if len(fragment) > 10:
            score += 0.10

    # ── URL quá dài ───────────────────────────────────────────────
    if len(u) > 100: score += 0.05
    if len(u) > 150: score += 0.07
    if len(u) > 250: score += 0.08

    # ── Mã hóa % quá nhiều (obfuscation) ─────────────────────────
    pct_encoded = len(re.findall(r"%[0-9a-fA-F]{2}", u))
    if pct_encoded >= 3:  score += 0.08
    if pct_encoded >= 8:  score += 0.10

    # ── Giả mạo brand trong domain ────────────────────────────────
    brands = [
        "paypal", "google", "facebook", "apple", "microsoft",
        "amazon", "netflix", "instagram", "twitter", "bank",
        "vietcombank", "momo", "techcombank", "bidv", "zalo",
        "mbbank", "vpbank", "agribank", "tpbank", "sacombank",
        "steam", "roblox", "binance", "coinbase",
    ]
    for brand in brands:
        if brand in domain and not _is_safe_domain(domain):
            score += 0.32
            break

    # ── Giả mạo brand trong path (khi domain khác) ───────────────
    if not hosting_hit:
        brand_in_path = sum(1 for b in brands if b in path)
        if brand_in_path >= 1:
            score += 0.08

    return min(score, 1.0)


def hybrid_score(dl_prob: float, url: str, dl_weight: float = 0.25) -> float:
    """
    Kết hợp DL probability với heuristic score.
    v3: DL 25% | Heuristic 75%
    Heuristic chiếm ưu thế vì dataset DL không cover phishing hiện đại
    (platform-abuse: Firebase, Weebly, Google Docs, URL shorteners)
    """
    h = heuristic_score(url)
    if h <= 0.05:
        # Domain uy tín + path sạch → kéo mạnh về SAFE
        return max(dl_prob * 0.08, h)
    if h >= 0.70:
        # Heuristic rất chắc là phishing → không để DL kéo xuống nhiều
        return min(0.80 * h + 0.20 * dl_prob, 1.0)
    return dl_weight * dl_prob + (1 - dl_weight) * h



def tokenize_urls(urls: list[str], tok: dict) -> "np.ndarray":
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    chars = tok["chars"]
    c2idx = {c: i + 2 for i, c in enumerate(chars)}
    max_len = tok["max_len"]
    seqs = [[c2idx.get(c, 1) for c in u.lower()[:max_len]] for u in urls]
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")



# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok" if _models else "degraded",
        models_loaded=list(_models.keys()),
        tokenizer_ready=_tokenizer is not None,
    )


@app.get("/models", tags=["System"])
async def list_models():
    return {"available": list(_models.keys()), "default": "CNN_BiLSTM"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    if not _tokenizer:
        raise HTTPException(503, detail="Tokenizer chưa được load. Kiểm tra tokenizer.json")
    if req.model_name not in _models:
        raise HTTPException(
            404,
            detail=f"Model '{req.model_name}' không tìm thấy. Available: {list(_models.keys())}"
        )
    if not req.url.strip():
        raise HTTPException(400, detail="URL không được để trống")

    model = _models[req.model_name]
    norm = normalize_url(req.url)
    X = tokenize_urls([norm], _tokenizer)
    dl_prob = float(model.predict(X, verbose=0)[0][0])
    prob = hybrid_score(dl_prob, norm)   # DL 55% + heuristic 45%
    is_phishing = prob >= req.threshold

    return PredictResponse(
        url=req.url,
        normalized_url=norm,
        probability=round(prob, 6),
        is_phishing=is_phishing,
        label="PHISHING" if is_phishing else "SAFE",
        confidence_pct=round((prob if is_phishing else 1 - prob) * 100, 2),
        model=req.model_name,
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    urls: list[str],
    model_name: str = "CNN_BiLSTM",
    threshold: float = 0.28,   # v3: đồng bộ với /predict
):
    if not _tokenizer or model_name not in _models:
        raise HTTPException(503, detail="Model/tokenizer chưa sẵn sàng")
    if not urls:
        raise HTTPException(400, detail="Danh sách URL trống")

    model = _models[model_name]
    normalized = [normalize_url(u) for u in urls]
    X = tokenize_urls(normalized, _tokenizer)
    dl_probs = model.predict(X, batch_size=256, verbose=0).flatten()
    probs = [hybrid_score(float(p), n) for p, n in zip(dl_probs, normalized)]

    return [
        {
            "url": u,
            "normalized_url": n,
            "probability": round(p, 6),
            "is_phishing": bool(p >= threshold),
            "label": "PHISHING" if p >= threshold else "SAFE",
        }
        for u, n, p in zip(urls, normalized, probs)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
