1. CHẠY HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH HỌC MÁY (ML):
Mở PowerShell và di chuyển vào thư mục:
  cd "d:\Đồ án\phishing_detector"
* Chạy đầy đủ dữ liệu (toàn bộ URL):
  main.py --csv phishing_urls_data.csv
* Chạy thử nghiệm nhanh (lấy mẫu 10,000 URL để xem kết quả nhanh):
  & "venv312\Scripts\python.exe" main.py --csv phishing_urls_data.csv --sample 10000

* Chạy với các file dữ liệu khác:
  main.py --csv Phishing.csv --sample 10000
  main.py --csv FineTune.csv
* Chạy TOÀN BỘ 500.000 URLs có kèm cả WHOIS + Tải trang web
  main.py --csv phishing_urls_data.csv --whois --fetch-page 

--------------------------------------------------

2. CHẠY GIAO DIỆN WEB DEMO PHÁT HIỆN LỪA ĐẢO:

Bước 1 - Chạy Backend API (Terminal 1):
  -m uvicorn api:app --port 8000

Bước 2 - Chạy Giao diện Streamlit (Terminal 2):
  run app.py

Bước 3 - Mở trình duyệt truy cập:
  http://localhost:8501
