"""
app.py — PhishGuard Cybersecurity Dashboard
Chạy: streamlit run app.py
"""
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(
    page_title="PhishGuard — Cybersecurity Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

html, body, .stApp { background: #080b14 !important; }

/* Global text */
*, p, span, div, label, h1, h2, h3 { color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
input, textarea {
    background: #0f1629 !important;
    color: #00d4ff !important;
    caret-color: #00d4ff !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.95em !important;
}
input:focus, textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
::placeholder { color: #2a4a6b !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #0f1629 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
/* Đổi màu nền tối và chữ vàng cho danh sách lựa chọn */
div[role="listbox"] ul, 
ul {
    background-color: #0f1629!important; /* Nền xanh tối navy */
    border: 1px solid #00d4ff!important; /* Viền xanh neon */
}

/* Định dạng màu chữ vàng cho từng mục lựa chọn */
div[role="listbox"] div, 
ul li[role="option"] {
    color: #FFD700!important; /* Chữ màu vàng */
    font-family: 'Share Tech Mono', monospace!important; /* Phông chữ công nghệ */
}

/* Hiệu ứng khi di chuột (hover) qua các lựa chọn */
div[role="listbox"] div:hover, 
ul li[role="option"]:hover {
    background-color: #1a233a!important; /* Nền sáng hơn một chút khi hover */
    color: #00d4ff!important; /* Chữ chuyển sang xanh neon khi hover */
}
/* Button */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0066ff) !important;
    color: #080b14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-size: 1em !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(0,212,255,0.55) !important;
}

/* Cards */
.card {
    background: rgba(15,22,41,0.8);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
}
.card-phishing {
    background: rgba(255,30,60,0.08);
    border: 1px solid rgba(255,30,60,0.4);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 0 30px rgba(255,30,60,0.15);
}
.card-safe {
    background: rgba(0,255,136,0.07);
    border: 1px solid rgba(0,255,136,0.35);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 0 30px rgba(0,255,136,0.12);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0e1a !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1e3a5f !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; background: transparent !important; }
.stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

/* Metric */
[data-testid="stMetric"] { background: rgba(15,22,41,0.8); border: 1px solid #1e3a5f; border-radius: 12px; padding: 16px; }

/* Hide defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────
def api_health():
    try:
        r = requests.get(f"{API}/health", timeout=2)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_predict(url: str, model: str, threshold: float) -> dict | None:
    try:
        r = requests.post(
            f"{API}/predict",
            json={"url": url, "model_name": model, "threshold": threshold},
            timeout=15,
        )
        if r.ok:
            return r.json()
        return {"error": r.json().get("detail", r.text)}
    except requests.ConnectionError:
        return {"error": "Không kết nối được API. Chạy api.py trước!"}
    except Exception as e:
        return {"error": str(e)}


def api_predict_batch(urls: list[str], model: str, threshold: float) -> list | None:
    try:
        r = requests.post(
            f"{API}/predict/batch",
            json=urls,
            params={"model_name": model, "threshold": threshold},
            timeout=60,
        )
        return r.json() if r.ok else None
    except Exception:
        return None


def gauge_chart(prob: float, is_phishing: bool) -> go.Figure:
    color = "#ff1e3c" if is_phishing else "#00ff88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Share Tech Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#0f1629",
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0, 40], "color": "rgba(0,255,136,0.08)"},
                {"range": [40, 60], "color": "rgba(255,187,0,0.08)"},
                {"range": [60, 100], "color": "rgba(255,30,60,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": round(prob * 100, 1)},
        },
        title={"text": "Phishing Probability", "font": {"size": 14, "color": "#64748b"}},
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#080b14",
        plot_bgcolor="#080b14",
        font_color="#e2e8f0",
    )
    return fig


# ── Session State ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 16px'>
        <div style='font-size:2.8em'>🛡️</div>
        <div style='font-size:1.2em; font-weight:700; color:#00d4ff !important;
                    font-family:"Share Tech Mono",monospace; letter-spacing:.08em'>
            PHISHGUARD
        </div>
        <div style='font-size:.75em; color:#334155 !important; margin-top:4px'>
            CYBERSECURITY DASHBOARD v2.0
        </div>
    </div>
    <hr style='border-color:#1e3a5f; margin:0 0 20px'/>
    """, unsafe_allow_html=True)

    # API status
    health = api_health()
    if health and health.get("models_loaded"):
        st.markdown(f"""
        <div style='background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);
                    border-radius:10px;padding:12px 16px;margin-bottom:16px'>
            <span style='color:#00ff88 !important;font-weight:600'>● API ONLINE</span><br/>
            <span style='color:#64748b !important;font-size:.8em'>
                Models: {", ".join(health["models_loaded"])}
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(255,30,60,0.08);border:1px solid rgba(255,30,60,0.3);
                    border-radius:10px;padding:12px 16px;margin-bottom:16px'>
            <span style='color:#ff1e3c !important;font-weight:600'>● API OFFLINE</span><br/>
            <span style='color:#64748b !important;font-size:.8em'>
                Chạy api.py để bắt đầu
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Cấu hình")
    model_name = st.selectbox(
        "Model DL:",
        ["CNN_BiLSTM", "CharCNN", "BiLSTM"],
        index=0,
        help="CNN_BiLSTM đạt F1=0.9716 cao nhất",
    )
    threshold = st.slider(
        "Ngưỡng phishing", 0.3, 0.9, 0.5, 0.05,
        help="Xác suất ≥ ngưỡng → phishing",
        format="%.2f",
    )

    st.markdown("---")
    st.markdown("#### 📊 Phiên này")
    total = len(st.session_state.history)
    phish_cnt = sum(1 for h in st.session_state.history if h["phishing"])
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Đã quét", total)
    with col_s2:
        st.metric("🔴 Phishing", phish_cnt)

    if total and st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72em;color:#334155 !important;text-align:center;padding:8px 0'>
        CharCNN · BiLSTM · CNN-BiLSTM<br/>Trained on phishing_urls_data.csv
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:28px'>
    <h1 style='font-family:"Share Tech Mono",monospace;font-size:2.2em;
               color:#00d4ff !important;letter-spacing:.06em;margin:0'>
        🛡️ PHISHGUARD
    </h1>
    <p style='color:#334155 !important;margin:4px 0 0;font-size:.95em'>
        Deep Learning · URL Phishing Detection · Real-time Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Kiểm tra URL", "📋 Kiểm tra hàng loạt", "📜 Lịch sử"])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — Single URL
# ════════════════════════════════════════════════════════════════════
with tab1:
    if not (health and health.get("models_loaded")):
        st.error("""
**⚠️ API chưa khởi động!**

Mở terminal mới và chạy:
```
d:\\Đồ án\\venv312\\Scripts\\python.exe -m uvicorn api:app --port 8000
```
""")

    url_input = st.text_input(
        "🔗 Nhập URL cần kiểm tra:",
        placeholder="Ví dụ: paypal-secure.login.xyz/verify hoặc https://google.com",
        key="url_single",
    )

    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        scan_btn = st.button(" Kiểm tra ", type="primary", use_container_width=True)
    with col_ex:
        st.caption("Hỗ trợ cả URL có/không có http://")

    if scan_btn:
        if not url_input.strip():
            st.warning("Vui lòng nhập URL!")
        else:
            with st.spinner("🔄 Đang phân tích..."):
                t0 = time.time()
                result = api_predict(url_input.strip(), model_name, threshold)
                elapsed = time.time() - t0

            if result and "error" not in result:
                is_phishing = result["is_phishing"]
                prob = result["probability"]

                # Result card
                if is_phishing:
                    st.markdown(f"""
                    <div class="card-phishing">
                        <div style='font-size:2em;font-weight:800;color:#ff1e3c !important;
                                    font-family:"Share Tech Mono",monospace;letter-spacing:.05em'>
                            🚨 PHISHING DETECTED
                        </div>
                        <div style='color:#94a3b8 !important;margin-top:8px;font-size:.95em'>
                            URL: <code style='color:#ff6b6b !important'>{result['normalized_url']}</code>
                        </div>
                        <div style='margin-top:12px;color:#94a3b8 !important;font-size:.88em'>
                            Model: {result['model']} &nbsp;·&nbsp; Thời gian: {elapsed:.2f}s
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card-safe">
                        <div style='font-size:2em;font-weight:800;color:#00ff88 !important;
                                    font-family:"Share Tech Mono",monospace;letter-spacing:.05em'>
                            ✅ SAFE
                        </div>
                        <div style='color:#94a3b8 !important;margin-top:8px;font-size:.95em'>
                            URL: <code style='color:#6ee7b7 !important'>{result['normalized_url']}</code>
                        </div>
                        <div style='margin-top:12px;color:#94a3b8 !important;font-size:.88em'>
                            Model: {result['model']} &nbsp;·&nbsp; Thời gian: {elapsed:.2f}s
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br/>", unsafe_allow_html=True)

                # Metrics + Gauge
                col_g, col_m = st.columns([1, 1])
                with col_g:
                    st.plotly_chart(gauge_chart(prob, is_phishing), use_container_width=True)
                with col_m:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    color = "#ff1e3c" if is_phishing else "#00ff88"
                    st.markdown(f"""
                    <div class="card" style='margin-top:8px'>
                        <div style='color:#64748b !important;font-size:.8em;text-transform:uppercase;letter-spacing:.06em'>
                            Phishing Probability
                        </div>
                        <div style='font-size:2.8em;font-weight:800;color:{color} !important;
                                    font-family:"Share Tech Mono",monospace;margin:6px 0'>
                            {prob*100:.2f}%
                        </div>
                        <hr style='border-color:#1e3a5f'/>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px'>
                            <div>
                                <div style='color:#64748b !important;font-size:.75em'>NGƯỠNG</div>
                                <div style='color:#e2e8f0 !important;font-weight:600'>{threshold:.0%}</div>
                            </div>
                            <div>
                                <div style='color:#64748b !important;font-size:.75em'>MODEL</div>
                                <div style='color:#e2e8f0 !important;font-weight:600'>{result['model']}</div>
                            </div>
                            <div>
                                <div style='color:#64748b !important;font-size:.75em'>KẾT QUẢ</div>
                                <div style='font-weight:700;color:{color} !important'>
                                    {"PHISHING" if is_phishing else "SAFE"}
                                </div>
                            </div>
                            <div>
                                <div style='color:#64748b !important;font-size:.75em'>LATENCY</div>
                                <div style='color:#e2e8f0 !important;font-weight:600'>{elapsed*1000:.0f}ms</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Save to history
                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "url": url_input.strip()[:60],
                    "model": result["model"],
                    "prob": f"{prob*100:.2f}%",
                    "phishing": is_phishing,
                    "result": "🔴 PHISHING" if is_phishing else "🟢 SAFE",
                })
                st.session_state.history = st.session_state.history[:50]

            elif result:
                st.error(f"❌ Lỗi: {result.get('error')}")


# ════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Scan
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📋 Kiểm tra nhiều URL cùng lúc")

    input_mode = st.radio(
        "Cách nhập:",
        ["✍️ Nhập tay (mỗi dòng 1 URL)", "📁 Upload file TXT/CSV"],
        horizontal=True,
    )

    urls_batch: list[str] = []

    if input_mode == "✍️ Nhập tay (mỗi dòng 1 URL)":
        raw = st.text_area(
            "Danh sách URL:",
            placeholder="https://google.com\nhttp://paypal-fake.xyz/login\nfacebook.com",
            height=160,
        )
        if raw.strip():
            urls_batch = [u.strip() for u in raw.splitlines() if u.strip()]
    else:
        up = st.file_uploader("Upload TXT hoặc CSV (cột 'url'):", type=["txt", "csv"])
        if up:
            if up.name.endswith(".csv"):
                _df = pd.read_csv(up)
                _df.columns = [c.lower().strip() for c in _df.columns]
                urls_batch = _df["url"].dropna().astype(str).tolist() if "url" in _df.columns else []
                if not urls_batch:
                    st.error("CSV cần có cột 'url'")
            else:
                urls_batch = [l.strip() for l in up.read().decode().splitlines() if l.strip()]

    if urls_batch:
        st.info(f"📋 Sẵn sàng kiểm tra **{len(urls_batch):,}** URL")

    batch_btn = st.button(" Kiểm tra ", type="primary", disabled=not urls_batch)

    if batch_btn and urls_batch:
        with st.spinner(f"Đang phân tích {len(urls_batch)} URL..."):
            results = api_predict_batch(urls_batch, model_name, threshold)

        if results:
            df_res = pd.DataFrame(results)
            df_res["Kết quả"] = df_res["is_phishing"].map({True: "🔴 PHISHING", False: "🟢 SAFE"})
            df_res["Xác suất"] = (df_res["probability"] * 100).round(2).astype(str) + "%"
            df_show = df_res[["url", "Xác suất", "Kết quả"]].rename(columns={"url": "URL"})

            phish_n = df_res["is_phishing"].sum()
            safe_n = len(df_res) - phish_n

            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng", len(df_res))
            c2.metric("🔴 Phishing", int(phish_n))
            c3.metric("🟢 Safe", int(safe_n))

            # Pie chart
            fig_pie = go.Figure(go.Pie(
                labels=["Phishing", "Safe"],
                values=[int(phish_n), int(safe_n)],
                marker_colors=["#ff1e3c", "#00ff88"],
                hole=0.5,
                textfont={"color": "#e2e8f0"},
            ))
            fig_pie.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#080b14", plot_bgcolor="#080b14",
                legend_font_color="#e2e8f0", showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(df_show, use_container_width=True, height=360)

            csv_out = df_show.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Tải kết quả CSV",
                csv_out, "batch_results.csv", "text/csv",
                use_container_width=True,
            )
        else:
            st.error("❌ Không lấy được kết quả từ API")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📜 Lịch sử kiểm tra trong phiên này")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        df_hist = df_hist[["time", "url", "model", "prob", "result"]]
        df_hist.columns = ["Thời gian", "URL", "Model", "Xác suất", "Kết quả"]
        st.dataframe(df_hist, use_container_width=True, height=500)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#334155 !important'>
            <div style='font-size:3em;margin-bottom:12px'>📭</div>
            <div>Chưa có lịch sử. Scan URL ở tab đầu tiên!</div>
        </div>""", unsafe_allow_html=True)
