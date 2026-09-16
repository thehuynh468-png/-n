import re
import math
import socket
import logging
from collections import Counter
from urllib.parse import urlparse, parse_qs

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False
    logging.warning("tldextract không khả dụng. Sẽ dùng phương pháp thay thế.")

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    logging.warning("python-whois không khả dụng. Bỏ qua đặc trưng host-based.")

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests/BeautifulSoup không khả dụng. Bỏ qua đặc trưng content-based.")

from datetime import datetime

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SENSITIVE_KEYWORDS = [
    "login", "signin", "sign-in", "bank", "secure", "account",
    "verify", "password", "paypal", "update", "confirm", "billing",
    "credential", "authenticate", "webscr", "logon",
    # Tiếng Việt
    "dangnhap", "dang-nhap", "thanhtoan", "thanh-toan",
    "baomat", "bao-mat", "taikhoan", "tai-khoan",
    "xacnhan", "xac-nhan", "matkhau", "mat-khau",
    "vietcombank", "techcombank", "bidv", "mbbank",
    "agribank", "vpbank", "hdbank", "sacombank", "acb",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "ow.ly", "goo.gl", "t.co",
    "is.gd", "cli.gs", "pic.gd", "yfrog.com", "migre.me",
    "ff.im", "tiny.cc", "url4.eu", "twurl.nl", "snipurl.com",
    "short.to", "BudURL.com", "ping.fm", "post.ly", "Just.as",
    "bkite.com", "snipr.com", "fic.kr", "loopt.us", "doiop.com",
    "short.ie", "kl.am", "wp.me", "rubyurl.com", "om.ly",
    "to.ly", "bit.do", "t2m.io", "qr.ae", "cutt.ly",
    "v.gd", "rb.gy", "shorte.st", "linktr.ee", "zws.im",
]

# TLD mien phi / it uy tin, pho bien trong phishing
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",       # mien phi tuyet doi
    "pw", "top", "xyz", "club", "online",
    "site", "website", "tech", "live",
    "click", "link", "download", "zip",
}

# Thuong hieu ngan hang / mang xa hoi lon hay bi gia mao
BRAND_KEYWORDS = [
    "vietcombank", "techcombank", "bidv", "mbbank", "agribank",
    "vpbank", "hdbank", "sacombank", "acb", "tpbank", "shinhan",
    "paypal", "apple", "google", "microsoft", "facebook", "amazon",
    "netflix", "instagram", "twitter", "linkedin", "dropbox",
]

# ─────────────────────────────────────────────
# Nhóm 1: Lexical Features
# ─────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    """Tinh entropy Shannon cua mot chuoi ky tu.
    URL ngau nhien (phishing) thuong co entropy cao hon URL hop le.
    """
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _get_domain_info(url: str):
    """Trả về (domain, suffix, subdomain) an toàn."""
    try:
        if TLDEXTRACT_AVAILABLE:
            ext = tldextract.extract(url)
            registered = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            return ext.domain, ext.suffix, ext.subdomain, registered
        else:
            parsed = urlparse(url)
            host = parsed.netloc.split(":")[0]
            parts = host.split(".")
            if len(parts) >= 2:
                domain = parts[-2]
                suffix = parts[-1]
                subdomain = ".".join(parts[:-2])
                registered = f"{domain}.{suffix}"
            else:
                domain = host
                suffix = ""
                subdomain = ""
                registered = host
            return domain, suffix, subdomain, registered
    except Exception:
        return "", "", "", ""


def extract_lexical_features(url: str) -> dict:
    """Trích xuất 18 đặc trưng lexical từ URL."""
    features = {}
    
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        parsed = urlparse("")

    # Các thành phần URL
    scheme   = parsed.scheme or ""
    netloc   = parsed.netloc or ""
    path     = parsed.path or ""
    query    = parsed.query or ""
    host     = netloc.split(":")[0] if netloc else ""

    domain, suffix, subdomain, registered_domain = _get_domain_info(url)

    # 1. url_length
    features["url_length"] = len(url)

    # 2. domain_length
    features["domain_length"] = len(registered_domain)

    # 3. num_dots
    features["num_dots"] = url.count(".")

    # 4. num_slashes  (chỉ tính trong path)
    features["num_slashes"] = url.count("/")

    # 5. num_question_marks
    features["num_question_marks"] = url.count("?")

    # 6. num_equal_signs
    features["num_equal_signs"] = url.count("=")

    # 7. num_hyphens
    features["num_hyphens"] = url.count("-")

    # 8. num_underscores
    features["num_underscores"] = url.count("_")

    # 9. num_at_signs
    features["num_at_signs"] = url.count("@")

    # 10. num_and_signs
    features["num_and_signs"] = url.count("&")

    # 11. path_depth
    path_parts = [p for p in path.split("/") if p]
    features["path_depth"] = len(path_parts)

    # 12. query_length
    features["query_length"] = len(query)

    # 13. num_query_params
    features["num_query_params"] = len(parse_qs(query)) if query else 0

    # 14. has_sensitive_keyword
    url_lower = url.lower()
    features["has_sensitive_keyword"] = int(
        any(kw in url_lower for kw in SENSITIVE_KEYWORDS)
    )

    # 15. has_ip_address
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    features["has_ip_address"] = int(bool(re.match(ipv4_pattern, host)))

    # 16. is_shortened
    features["is_shortened"] = int(
        any(shortener in host.lower() for shortener in URL_SHORTENERS)
    )

    # 17. uses_https
    features["uses_https"] = int(scheme.lower() == "https")

    # 18. num_subdomains
    if subdomain and subdomain.lower() != "www":
        features["num_subdomains"] = subdomain.count(".") + 1
    elif subdomain and subdomain.lower() == "www":
        features["num_subdomains"] = 0
    else:
        features["num_subdomains"] = 0

    # ── Đặc trưng mới (v2) ── #

    # 19. has_suspicious_tld: TLD thuoc danh sach mien phi / it uy tin
    features["has_suspicious_tld"] = int(suffix.lower() in SUSPICIOUS_TLDS)

    # 20. digit_ratio: ty le ky tu so trong toan bo URL
    digits_in_url = sum(c.isdigit() for c in url)
    features["digit_ratio"] = round(digits_in_url / max(len(url), 1), 4)

    # 21. has_encoded_chars: co chuoi ma hoa %XX trong URL khong
    features["has_encoded_chars"] = int(bool(re.search(r"%[0-9A-Fa-f]{2}", url)))

    # 22. url_entropy: entropy Shannon cua URL (cao = ngau nhien = kha nang phishing)
    features["url_entropy"] = round(_shannon_entropy(url), 4)

    # 23. num_digits_in_domain: so luong ky tu so trong ten mien chinh
    features["num_digits_in_domain"] = sum(c.isdigit() for c in domain)

    # 24. path_token_count: so luong token trong path (tach theo / - _)
    path_tokens = re.split(r"[/\-_]", path)
    features["path_token_count"] = len([t for t in path_tokens if t])

    # 25. has_brand_in_subdomain: brand xuat hien trong subdomain nhung khong trong domain
    #     Dau hieu gia mao: vietcombank.evil.com
    brand_in_sub    = any(b in subdomain.lower() for b in BRAND_KEYWORDS) if subdomain else False
    brand_in_domain = any(b in domain.lower()    for b in BRAND_KEYWORDS)
    features["has_brand_in_subdomain"] = int(brand_in_sub and not brand_in_domain)

    # 26. has_redirect: URL chua 'http' lan thu 2 (ky thuat redirect che giau)
    features["has_redirect"] = int(url.lower().count("http") > 1)

    return features


# ─────────────────────────────────────────────
# Nhóm 2: Host-based Features (WHOIS)
# ─────────────────────────────────────────────

def extract_host_features(url: str) -> dict:
    """Tra cứu WHOIS để lấy tuổi domain và privacy status."""
    features = {"domain_age_days": -1, "whois_private": -1}

    if not WHOIS_AVAILABLE:
        return features

    try:
        _, _, _, registered_domain = _get_domain_info(url)
        if not registered_domain:
            return features

        w = whois.whois(registered_domain)

        # domain_age_days
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            if isinstance(creation_date, datetime):
                age = (datetime.now() - creation_date).days
                features["domain_age_days"] = max(age, 0)

        # whois_private
        org = str(w.org or "").lower()
        name = str(w.name or "").lower()
        privacy_keywords = ["privacy", "private", "protect", "redacted", "whoisguard"]
        features["whois_private"] = int(
            any(kw in org or kw in name for kw in privacy_keywords)
        )
    except Exception as e:
        logging.debug(f"WHOIS lookup failed for {url}: {e}")

    return features


# ─────────────────────────────────────────────
# Nhóm 3: Content-based Features (HTTP fetch)
# ─────────────────────────────────────────────

def extract_content_features(url: str, timeout: int = 5) -> dict:
    """Tải trang web và kiểm tra nội dung."""
    features = {"has_password_form": 0, "page_title": ""}

    if not REQUESTS_AVAILABLE:
        return features

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")

        # has_password_form
        pwd_inputs = soup.find_all("input", {"type": "password"})
        features["has_password_form"] = int(len(pwd_inputs) > 0)

        # page_title
        title_tag = soup.find("title")
        features["page_title"] = title_tag.get_text(strip=True) if title_tag else ""

    except Exception as e:
        logging.debug(f"Content fetch failed for {url}: {e}")

    return features


# ─────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────

def extract_features(
    url: str,
    fetch_whois: bool = False,
    fetch_page: bool = False,
) -> dict:
    """
    Trích xuất toàn bộ đặc trưng từ một URL.
    
    Args:
        url: URL cần phân tích.
        fetch_whois: Tra cứu WHOIS (chậm, ~2-5s/URL).
        fetch_page: Tải trang web (chậm, ~3-10s/URL).
    
    Returns:
        Dictionary chứa tất cả đặc trưng đã trích xuất.
    """
    features = {}
    features.update(extract_lexical_features(url))

    if fetch_whois:
        features.update(extract_host_features(url))
    else:
        features["domain_age_days"] = -1
        features["whois_private"] = -1

    if fetch_page:
        features.update(extract_content_features(url))
    else:
        features["has_password_form"] = 0
        features["page_title"] = ""

    return features


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Column groups (Theo đúng cấu trúc bài báo: X1, X2, X3)
# ─────────────────────────────────────────────

# X1: Lexical Features (26 đặc trưng từ chuỗi URL)
LEXICAL_COLS = [
    # --- Nhóm đặc trưng cơ bản ---
    "url_length", "domain_length", "num_dots", "num_slashes",
    "num_question_marks", "num_equal_signs", "num_hyphens",
    "num_underscores", "num_at_signs", "num_and_signs",
    "path_depth", "query_length", "num_query_params",
    "has_sensitive_keyword", "has_ip_address", "is_shortened",
    "uses_https", "num_subdomains",
    # --- Đặc trưng mở rộng ---
    "has_suspicious_tld", "digit_ratio", "has_encoded_chars",
    "url_entropy", "num_digits_in_domain", "path_token_count",
    "has_brand_in_subdomain", "has_redirect",
]

# X2: Host-based Features (WHOIS)
HOST_COLS = ["domain_age_days", "whois_private"]

# X3: Content-based Features (HTTP/HTML)
CONTENT_COLS = ["has_password_form"]  # text column 'page_title' excluded from ML


if __name__ == "__main__":
    # Quick test
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login?user=admin&pass=1234",
        "http://bit.ly/3xPhIsH",
        "https://vietcombank-secure-login.com/dangnhap/xacnhan?acc=123&pin=456",
        "https://paypal.com-secure-update.phishing.tk/verify/account",
    ]
    import json
    for u in test_urls:
        f = extract_features(u)
        print(f"\nURL: {u}")
        print(json.dumps({k: v for k, v in f.items() if k != "page_title"}, indent=2))
