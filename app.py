import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from flask import jsonify, request
import plotly.express as px
import plotly.io as pio
import base64

# Chart types available for selection
CHART_TYPES = [
    'Bar Chart', 'Column Chart', 'Line Chart', 'Pie Chart', 'Donut Chart', 'Scatter Plot', 'Bubble Chart',
    'Histogram', 'Box Plot', 'Heatmap', 'Area Chart', 'Stacked Area Chart', 'Radar Chart', 'Funnel Chart', 'Treemap'
]

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['uploaded_file'] = filename
            return redirect(url_for('preview'))
        else:
            flash('Allowed file types are csv, xlsx')
            return redirect(request.url)
    return render_template('home.html')

@app.route('/preview')
def preview():
    filename = session.get('uploaded_file')
    if not filename:
        flash('No file uploaded yet.')
        return redirect(url_for('home'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    preview_data = df.head(20).to_html(classes='table table-striped', index=False)
    stats = df.describe(include='all').to_html(classes='table table-bordered', index=True)
    return render_template('preview.html', table=preview_data, stats=stats, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    filename = session.get('uploaded_file')
    if not filename:
        flash('No file uploaded yet.')
        return redirect(url_for('home'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    columns = df.columns.tolist()
    filter_col = group_col = agg_func = y_col = None
    chart_url = None
    top_n = 10
    orig_group_col = None
    if request.method == 'POST':
        filter_col = request.form.get('filter_col')
        group_col = request.form.get('group_col')
        agg_func = request.form.get('agg_func')
        y_col = request.form.get('y_col')
        top_n = int(request.form.get('top_n', 10))
        dff = df.copy()
        if filter_col and filter_col in dff.columns:
            dff = dff.dropna(subset=[filter_col])
        if group_col and group_col in dff.columns and y_col and y_col in dff.columns:
            orig_group_col = group_col
            if agg_func == 'sum':
                grouped = dff.groupby(group_col)[y_col].sum()
            elif agg_func == 'mean':
                grouped = dff.groupby(group_col)[y_col].mean()
            else:
                grouped = dff.groupby(group_col)[y_col].count()
            # Top N logic
            top_categories = grouped.sort_values(ascending=False).head(top_n)
            dff['__grouped__'] = dff[group_col].apply(lambda x: x if x in top_categories.index else 'Other')
            if agg_func == 'sum':
                grouped = dff.groupby('__grouped__')[y_col].sum()
            elif agg_func == 'mean':
                grouped = dff.groupby('__grouped__')[y_col].mean()
            else:
                grouped = dff.groupby('__grouped__')[y_col].count()
            chart_url = url_for('chart', group_col='__grouped__', agg_func=agg_func, y_col=y_col, top_n=top_n, orig_group_col=orig_group_col)
        else:
            grouped = dff
        session['analyze_params'] = {
            'filter_col': filter_col,
            'group_col': group_col,
            'agg_func': agg_func,
            'y_col': y_col,
            'top_n': top_n,
            'orig_group_col': orig_group_col
        }
    else:
        grouped = df
    return render_template('analyze.html', filename=filename, columns=columns, filter_col=filter_col, group_col=group_col, agg_func=agg_func, y_col=y_col, chart_url=chart_url)

@app.route('/chart')
def chart():
    filename = session.get('uploaded_file')
    if not filename:
        return '', 404
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    group_col = request.args.get('group_col')
    agg_func = request.args.get('agg_func')
    y_col = request.args.get('y_col')
    top_n = int(request.args.get('top_n', 10))
    orig_group_col = request.args.get('orig_group_col')
    # If group_col is __grouped__, use orig_group_col for Top N logic
    if group_col == '__grouped__' and orig_group_col and orig_group_col in df.columns and y_col and y_col in df.columns:
        if agg_func == 'sum':
            grouped = df.groupby(orig_group_col)[y_col].sum()
        elif agg_func == 'mean':
            grouped = df.groupby(orig_group_col)[y_col].mean()
        else:
            grouped = df.groupby(orig_group_col)[y_col].count()
        top_categories = grouped.sort_values(ascending=False).head(top_n)
        df['__grouped__'] = df[orig_group_col].apply(lambda x: x if x in top_categories.index else 'Other')
        if agg_func == 'sum':
            grouped = df.groupby('__grouped__')[y_col].sum()
        elif agg_func == 'mean':
            grouped = df.groupby('__grouped__')[y_col].mean()
        else:
            grouped = df.groupby('__grouped__')[y_col].count()
        fig, ax = plt.subplots(figsize=(10,5))
        grouped.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        return send_file(img, mimetype='image/png')
    elif group_col and group_col in df.columns and y_col and y_col in df.columns:
        if agg_func == 'sum':
            grouped = df.groupby(group_col)[y_col].sum()
        elif agg_func == 'mean':
            grouped = df.groupby(group_col)[y_col].mean()
        else:
            grouped = df.groupby(group_col)[y_col].count()
        top_categories = grouped.sort_values(ascending=False).head(top_n)
        df['__grouped__'] = df[group_col].apply(lambda x: x if x in top_categories.index else 'Other')
        if agg_func == 'sum':
            grouped = df.groupby('__grouped__')[y_col].sum()
        elif agg_func == 'mean':
            grouped = df.groupby('__grouped__')[y_col].mean()
        else:
            grouped = df.groupby('__grouped__')[y_col].count()
        fig, ax = plt.subplots(figsize=(10,5))
        grouped.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        return send_file(img, mimetype='image/png')
    return '', 404

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    filename = session.get('uploaded_file')
    if not filename:
        flash('No file uploaded yet.')
        return redirect(url_for('home'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    columns = df.columns.tolist()
    stats = df.describe(include='all').to_dict()
    if request.method == 'POST':
        chart_types = request.form.getlist('chart_type')
        x_col = request.form.get('x_col')
        y_col = request.form.get('y_col')
        group_col = request.form.get('group_col')
        size_col = request.form.get('size_col')
        color_col = request.form.get('color_col')
        style = request.form.get('style')
        top_n = int(request.form.get('top_n', 10))
        orientation = request.form.get('orientation', 'v')
        session['chart_params'] = {
            'chart_types': chart_types,
            'x_col': x_col,
            'y_col': y_col,
            'group_col': group_col,
            'size_col': size_col,
            'color_col': color_col,
            'style': style,
            'top_n': top_n,
            'orientation': orientation
        }
        return redirect(url_for('chart_view'))
    return render_template('visualize.html', filename=filename, columns=columns, stats=stats, chart_types=CHART_TYPES)

@app.route('/chart-view')
def chart_view():
    filename = session.get('uploaded_file')
    if not filename:
        flash('No file uploaded yet.')
        return redirect(url_for('home'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    params = session.get('chart_params', {})
    chart_types = params.get('chart_types', [])
    x_col = params.get('x_col')
    y_col = params.get('y_col')
    group_col = params.get('group_col')
    size_col = params.get('size_col')
    color_col = params.get('color_col')
    style = params.get('style', 'interactive')
    top_n = int(params.get('top_n', 10))
    orientation = params.get('orientation', 'v')
    chart_info = {
        'Chart Types': ', '.join(chart_types),
        'X': x_col,
        'Y': y_col,
        'Group': group_col,
        'Size': size_col,
        'Color': color_col,
        'Style': style,
        'Top N': top_n,
        'Orientation': 'Horizontal' if orientation == 'h' else 'Vertical'
    }
    chart_blocks = []
    for chart_type in chart_types:
        try:
            if style == 'interactive':
                fig = generate_plotly_chart(df, chart_type, x_col, y_col, group_col, size_col, color_col, orientation, top_n)
                chart_html = pio.to_html(fig, full_html=False)
                chart_blocks.append({'type': chart_type, 'html': chart_html, 'url': None, 'error': None})
            else:
                img = generate_matplotlib_chart(df, chart_type, x_col, y_col, group_col, size_col, color_col, orientation)
                chart_url = 'data:image/png;base64,' + base64.b64encode(img.getvalue()).decode()
                chart_blocks.append({'type': chart_type, 'html': None, 'url': chart_url, 'error': None})
        except Exception as e:
            chart_blocks.append({'type': chart_type, 'html': None, 'url': None, 'error': str(e)})
    return render_template('chart.html', filename=filename, chart_info=chart_info, chart_blocks=chart_blocks)

@app.route('/drilldown', methods=['POST'])
def drilldown():
    data = request.get_json()
    chart_type = data.get('chart_type')
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    group_col = data.get('group_col')
    top_n = int(data.get('top_n', 10))
    orientation = data.get('orientation', 'v')
    style = data.get('style', 'interactive')
    other_categories = data.get('other_categories')
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify({'chart_html': '<div class="alert alert-danger">No file uploaded.</div>'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    # Find categories in 'Other'
    if x_col not in df.columns or y_col not in df.columns:
        return jsonify({'chart_html': '<div class="alert alert-danger">Invalid columns for drilldown.</div>'})
    agg = df.groupby(x_col)[y_col].sum()
    top = agg.sort_values(ascending=False).head(top_n)
    other_mask = ~df[x_col].isin(top.index)
    other_df = df[other_mask]
    # For Pie/Donut: show breakdown of 'Other' categories
    if chart_type in ['Pie Chart', 'Donut Chart']:
        pie_df = other_df[[x_col, y_col]].dropna()
        pie_df = pie_df.groupby(x_col)[y_col].sum().reset_index()
        if pie_df.empty:
            chart_html = '<div class="alert alert-info">No further breakdown for Other.</div>'
        else:
            import plotly.express as px
            fig = px.pie(pie_df, names=x_col, values=y_col, hole=0.5 if chart_type == 'Donut Chart' else 0)
            import plotly.io as pio
            chart_html = pio.to_html(fig, full_html=False)
    # For Bar/Column: show breakdown of 'Other' categories
    elif chart_type in ['Bar Chart', 'Column Chart']:
        bar_df = other_df[[x_col, y_col]].dropna()
        bar_df = bar_df.groupby(x_col)[y_col].sum().reset_index()
        if bar_df.empty:
            chart_html = '<div class="alert alert-info">No further breakdown for Other.</div>'
        else:
            import plotly.express as px
            fig = px.bar(bar_df, x=x_col if orientation=='v' else y_col, y=y_col if orientation=='v' else x_col, orientation=orientation)
            import plotly.io as pio
            chart_html = pio.to_html(fig, full_html=False)
    else:
        chart_html = '<div class="alert alert-info">Drilldown not supported for this chart type.</div>'
    return jsonify({'chart_html': chart_html})

# Helper functions for chart generation (to be implemented)
def generate_plotly_chart(df, chart_type, x_col, y_col, group_col, size_col, color_col, orientation='v', top_n=10):
    import plotly.express as px
    # Sanitize optional columns
    if not color_col or color_col == 'None':
        color_col = None
    if not size_col or size_col == 'None':
        size_col = None
    if not group_col or group_col == 'None':
        group_col = None
    # Defensive: check columns
    if x_col and x_col not in df.columns:
        x_col = None
    if y_col and y_col not in df.columns:
        y_col = None
    if group_col and group_col not in df.columns:
        group_col = None
    if size_col and size_col not in df.columns:
        size_col = None
    if color_col and color_col not in df.columns:
        color_col = None
    # Chart logic
    if chart_type == 'Bar Chart' or chart_type == 'Column Chart':
        return px.bar(df, x=x_col if orientation=='v' else y_col, y=y_col if orientation=='v' else x_col, color=group_col or color_col, orientation=orientation, barmode='group')
    elif chart_type == 'Line Chart':
        return px.line(df, x=x_col, y=y_col, color=group_col or color_col)
    elif chart_type == 'Pie Chart' or chart_type == 'Donut Chart':
        if not x_col or x_col not in df.columns or not y_col or y_col not in df.columns:
            raise ValueError("Please select valid columns for both X (names) and Y (values) for Pie/Donut charts.")
        pie_df = df[[x_col, y_col]].dropna()
        if pie_df.empty or not pie_df[y_col].apply(lambda v: isinstance(v, (int, float)) and v != 0).any():
            raise ValueError("No valid or nonzero data to plot for Pie/Donut chart. Please select different columns or check your data.")
        # Top N + Other logic
        agg = pie_df.groupby(x_col)[y_col].sum()
        top = agg.sort_values(ascending=False).head(top_n)
        pie_df['Category'] = pie_df[x_col].apply(lambda x: x if x in top.index else 'Other')
        pie_df = pie_df.groupby('Category')[y_col].sum().reset_index()
        # Ensure only one 'Other' entry and no '__grouped__' in legend
        return px.pie(pie_df, names='Category', values=y_col, hole=0.5 if chart_type == 'Donut Chart' else 0)
    elif chart_type == 'Donut Chart':
        return px.pie(df, names=x_col, values=y_col, color=color_col, hole=0.5)
    elif chart_type == 'Scatter Plot':
        return px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, symbol=group_col)
    elif chart_type == 'Bubble Chart':
        return px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, symbol=group_col)
    elif chart_type == 'Histogram':
        return px.histogram(df, x=x_col, color=group_col or color_col)
    elif chart_type == 'Box Plot':
        return px.box(df, x=group_col or x_col, y=y_col, color=color_col)
    elif chart_type == 'Heatmap':
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Please select valid columns for both X and Y for Heatmap.")
        # Top N for X and Y
        x_counts = df[x_col].value_counts().head(top_n).index
        y_counts = df[y_col].value_counts().head(top_n).index
        heatmap_df = df[df[x_col].isin(x_counts) & df[y_col].isin(y_counts)]
        if heatmap_df.empty:
            raise ValueError("No data to plot for Heatmap after applying Top N filter. Try different columns or increase Top N.")
        return px.density_heatmap(heatmap_df, x=x_col, y=y_col, z=size_col, color_continuous_scale='Viridis')
    elif chart_type == 'Area Chart':
        return px.area(df, x=x_col, y=y_col, color=group_col or color_col)
    elif chart_type == 'Stacked Area Chart':
        return px.area(df, x=x_col, y=y_col, color=group_col or color_col, groupnorm='fraction', stackgroup='one')
    elif chart_type == 'Radar Chart':
        # Radar needs data in wide format
        import plotly.graph_objects as go
        if group_col and y_col:
            categories = list(df[x_col].unique()) if x_col else list(df.columns)
            fig = go.Figure()
            for name, group in df.groupby(group_col):
                fig.add_trace(go.Scatterpolar(r=group[y_col], theta=categories, fill='toself', name=str(name)))
            return fig
        else:
            return px.line_polar(df, r=y_col, theta=x_col, line_close=True)
    elif chart_type == 'Funnel Chart':
        import plotly.graph_objects as go
        if x_col and y_col:
            return go.Figure(go.Funnel(y=df[x_col], x=df[y_col]))
        else:
            return px.bar(df, x=x_col, y=y_col)
    elif chart_type == 'Treemap':
        return px.treemap(df, path=[group_col, x_col] if group_col else [x_col], values=y_col, color=color_col)
    else:
        return px.scatter(df, x=x_col, y=y_col)

def generate_matplotlib_chart(df, chart_type, x_col, y_col, group_col, size_col, color_col, orientation='v'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import io
    # Sanitize optional columns
    if not color_col or color_col == 'None':
        color_col = None
    if not size_col or size_col == 'None':
        size_col = None
    if not group_col or group_col == 'None':
        group_col = None
    fig, ax = plt.subplots(figsize=(10,5))
    # Defensive: check columns
    if x_col and x_col not in df.columns:
        x_col = None
    if y_col and y_col not in df.columns:
        y_col = None
    if group_col and group_col not in df.columns:
        group_col = None
    if size_col and size_col not in df.columns:
        size_col = None
    if color_col and color_col not in df.columns:
        color_col = None
    try:
        if chart_type == 'Bar Chart' or chart_type == 'Column Chart':
            if group_col:
                sns.barplot(data=df, x=x_col if orientation=='v' else y_col, y=y_col if orientation=='v' else x_col, hue=group_col, ax=ax, orient='v' if orientation=='v' else 'h')
            else:
                sns.barplot(data=df, x=x_col if orientation=='v' else y_col, y=y_col if orientation=='v' else x_col, ax=ax, orient='v' if orientation=='v' else 'h')
        elif chart_type == 'Line Chart':
            if group_col:
                sns.lineplot(data=df, x=x_col, y=y_col, hue=group_col, ax=ax)
            else:
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == 'Pie Chart' or chart_type == 'Donut Chart':
            vals = df[y_col].value_counts() if not x_col else df.groupby(x_col)[y_col].sum()
            labels = vals.index
            wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.5 if chart_type=='Donut Chart' else 1))
            ax.axis('equal')
        elif chart_type == 'Scatter Plot' or chart_type == 'Bubble Chart':
            sizes = df[size_col]*100 if size_col else None
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, size=size_col, ax=ax, sizes=(20, 200))
        elif chart_type == 'Histogram':
            sns.histplot(data=df, x=x_col, hue=group_col, ax=ax, bins=20)
        elif chart_type == 'Box Plot':
            sns.boxplot(data=df, x=group_col or x_col, y=y_col, hue=color_col, ax=ax)
        elif chart_type == 'Heatmap':
            if x_col and y_col:
                pt = pd.pivot_table(df, values=size_col or y_col, index=y_col, columns=x_col, aggfunc=np.mean)
                sns.heatmap(pt, ax=ax, cmap='viridis')
            else:
                sns.heatmap(df.corr(), ax=ax, cmap='viridis', annot=True)
        elif chart_type == 'Area Chart' or chart_type == 'Stacked Area Chart':
            if group_col:
                df_pivot = df.pivot_table(index=x_col, columns=group_col, values=y_col, aggfunc='sum').fillna(0)
                df_pivot.plot.area(ax=ax, stacked=(chart_type=='Stacked Area Chart'))
            else:
                df.plot.area(x=x_col, y=y_col, ax=ax)
        elif chart_type == 'Radar Chart':
            # Radar needs data in wide format
            categories = list(df[x_col].unique()) if x_col else list(df.columns)
            values = df[y_col].values if y_col else df.iloc[0].values
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values = np.concatenate((values, [values[0]]))
            angles += angles[:1]
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles), categories)
        elif chart_type == 'Funnel Chart':
            vals = df[y_col].values if y_col else df.iloc[:,0].values
            labels = df[x_col].values if x_col else df.index
            ax.barh(labels, vals)
        elif chart_type == 'Treemap':
            # Matplotlib doesn't have native treemap, use squarify if available
            try:
                import squarify
                vals = df[y_col] if y_col else df.iloc[:,0]
                squarify.plot(sizes=vals, label=df[x_col] if x_col else df.index, ax=ax)
            except ImportError:
                ax.text(0.5, 0.5, 'Install squarify for treemap', ha='center')
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        plt.xticks(rotation=45 if orientation=='v' else 0, ha='right')
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', ha='center')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

@app.route('/export')
def export():
    filename = session.get('uploaded_file')
    if not filename:
        flash('No file uploaded yet.')
        return redirect(url_for('home'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    params = session.get('analyze_params', {})
    group_col = params.get('group_col')
    agg_func = params.get('agg_func')
    y_col = params.get('y_col')
    top_n = params.get('top_n', 10)
    orig_group_col = params.get('orig_group_col')
    if group_col and group_col in df.columns and y_col and y_col in df.columns:
        if agg_func == 'sum':
            grouped = df.groupby(group_col)[y_col].sum(numeric_only=True)
        elif agg_func == 'mean':
            grouped = df.groupby(group_col)[y_col].mean(numeric_only=True)
        else:
            grouped = df.groupby(group_col)[y_col].count()
        # Top N logic for export
        top_categories = grouped.sort_values(ascending=False).head(top_n)
        df['__grouped__'] = df[group_col].apply(lambda x: x if x in top_categories.index else 'Other')
        if agg_func == 'sum':
            grouped = df.groupby('__grouped__')[y_col].sum()
        elif agg_func == 'mean':
            grouped = df.groupby('__grouped__')[y_col].mean()
        else:
            grouped = df.groupby('__grouped__')[y_col].count()
    elif orig_group_col and orig_group_col in df.columns and y_col and y_col in df.columns:
        if agg_func == 'sum':
            grouped = df.groupby(orig_group_col)[y_col].sum(numeric_only=True)
        elif agg_func == 'mean':
            grouped = df.groupby(orig_group_col)[y_col].mean(numeric_only=True)
        else:
            grouped = df.groupby(orig_group_col)[y_col].count()
        # Top N logic for export
        top_categories = grouped.sort_values(ascending=False).head(top_n)
        df['__grouped__'] = df[orig_group_col].apply(lambda x: x if x in top_categories.index else 'Other')
        if agg_func == 'sum':
            grouped = df.groupby('__grouped__')[y_col].sum()
        elif agg_func == 'mean':
            grouped = df.groupby('__grouped__')[y_col].mean()
        else:
            grouped = df.groupby('__grouped__')[y_col].count()
    else:
        grouped = df
    export_type = request.args.get('type', 'csv')
    if export_type == 'csv':
        out = io.StringIO()
        grouped.to_csv(out)
        out.seek(0)
        return send_file(io.BytesIO(out.read().encode()), mimetype='text/csv', as_attachment=True, download_name='export.csv')
    elif export_type == 'excel':
        out = io.BytesIO()
        grouped.to_excel(out)
        out.seek(0)
        return send_file(out, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='export.xlsx')
    elif export_type == 'png':
        fig, ax = plt.subplots(figsize=(8,4))
        grouped.plot(kind='bar', ax=ax)
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        return send_file(img, mimetype='image/png', as_attachment=True, download_name='chart.png')
    else:
        flash('Invalid export type.')
        return redirect(url_for('analyze'))

if __name__ == '__main__':
    app.run(debug=True) 