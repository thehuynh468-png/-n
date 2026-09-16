== HƯỚNG DẪN CHẠY DỰ ÁN PHISHGUARD ==

1. CHẠY HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH HỌC MÁY (ML):
Mở PowerShell và di chuyển vào thư mục:
  cd "d:\Đồ án\phishing_detector"

* Chạy đầy đủ dữ liệu (toàn bộ URL):
  & "d:\Đồ án\venv312\Scripts\python.exe" main.py --csv phishing_urls_data.csv

* Chạy thử nghiệm nhanh (lấy mẫu 10,000 URL để xem kết quả nhanh):
  & "d:\Đồ án\venv312\Scripts\python.exe" main.py --csv phishing_urls_data.csv --sample 10000

* Chạy với các file dữ liệu khác:
  & "d:\Đồ án\venv312\Scripts\python.exe" main.py --csv Phishing.csv --sample 10000
  & "d:\Đồ án\venv312\Scripts\python.exe" main.py --csv FineTune.csv

* Kết quả sau khi chạy:
  - Bảng so sánh Accuracy, Precision, Recall, F1-Score in ra màn hình.
  - Tự động xuất file báo cáo Word: BaoCao_Phishing_ML.docx

--------------------------------------------------

2. CHẠY GIAO DIỆN WEB DEMO PHÁT HIỆN LỪA ĐẢO:

Bước 1 - Chạy Backend API (Terminal 1):
  cd "d:\Đồ án\phishing_detector"
  & "d:\Đồ án\venv312\Scripts\python.exe" -m uvicorn api:app --port 8000

Bước 2 - Chạy Giao diện Streamlit (Terminal 2):
  cd "d:\Đồ án\phishing_detector"
  & "d:\Đồ án\venv312\Scripts\streamlit.exe" run app.py

Bước 3 - Mở trình duyệt truy cập:
  http://localhost:8501