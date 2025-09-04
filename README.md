# full-stack-flask-application


🚀 Overview
Data Analysis Dashboard is a modern, interactive web app built with Python and its powerful data stack. Effortlessly upload, analyze, visualize, and export your data—all in your browser!


🧰 Python Libraries Used


Flask: Web framework for backend routing and API.
Pandas: Data wrangling, filtering, grouping, and statistics.
NumPy: Fast numerical operations.
Matplotlib, Seaborn: Static chart generation.
Plotly: Interactive, beautiful charts (bar, pie, scatter, heatmap, etc.).
openpyxl: Excel file support.
gunicorn: Production WSGI server.
flask-caching: Speed up heavy queries (optional).



✨ Features

1) Upload CSV or Excel files
2) Preview data and summary stats
3) Filter, group, and aggregate with Pandas
4) Visualize with 15+ chart types (Bar, Pie, Line, Heatmap, etc.)
5) Drilldown: Click "Other" in Pie/Bar charts for deeper insights
6) Export filtered data and charts (CSV, Excel, PNG)
7) Modern UI: Responsive, Bootstrap 5, loading animations
8) No page reloads: AJAX/Plotly for smooth interactivity







⚡ Quickstart
# 1. Clone the repo
git clone https://github.com/HassanCodesIt/Data-Analysis-Dashboard.git
cd Data-Analysis-Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
python app.py

# 4. Or run in production
gunicorn -w 4 app:app

# 5. Open in your browser
# http://127.0.0.1:5000/
