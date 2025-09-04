# Data Analysis Dashboard 🐍📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-yellow?logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-blueviolet?logo=plotly)](https://plotly.com/python/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.x-purple?logo=bootstrap)](https://getbootstrap.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-green?logo=render)](https://data-analysis-dashboard-w47o.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Overview

**Data Analysis Dashboard** is a modern, interactive web app built with Python and its powerful data stack. Effortlessly upload, analyze, visualize, and export your data—all in your browser!

---

## 🧰 Python Libraries Used

- **[Flask](https://flask.palletsprojects.com/):** Web framework for backend routing and API.
- **[Pandas](https://pandas.pydata.org/):** Data wrangling, filtering, grouping, and statistics.
- **[NumPy](https://numpy.org/):** Fast numerical operations.
- **[Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/):** Static chart generation.
- **[Plotly](https://plotly.com/python/):** Interactive, beautiful charts (bar, pie, scatter, heatmap, etc.).
- **[openpyxl](https://openpyxl.readthedocs.io/):** Excel file support.
- **[gunicorn](https://gunicorn.org/):** Production WSGI server.
- **[flask-caching](https://flask-caching.readthedocs.io/):** Speed up heavy queries (optional).

---

## ✨ Features

- **Upload** CSV or Excel files
- **Preview** data and summary stats
- **Filter, group, and aggregate** with Pandas
- **Visualize** with 15+ chart types (Bar, Pie, Line, Heatmap, etc.)
- **Drilldown:** Click "Other" in Pie/Bar charts for deeper insights
- **Export** filtered data and charts (CSV, Excel, PNG)
- **Modern UI:** Responsive, Bootstrap 5, loading animations
- **No page reloads:** AJAX/Plotly for smooth interactivity

---

## ⚡ Quickstart

```bash
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
```

---

## 🌐 Live Demo

Try it now: [data-analysis-dashboard-w47o.onrender.com](https://data-analysis-dashboard-w47o.onrender.com/)

---

## 🧑‍💻 For Developers

- Modular Flask app (`app.py`)
- All templates in `/templates/`
- Static assets in `/static/`
- Uploads in `/uploads/`
- Easily extend with new chart types or data sources

---

## 📄 License

MIT

---

> **Built with Python, Pandas, and Plotly for data lovers and analysts!** 
