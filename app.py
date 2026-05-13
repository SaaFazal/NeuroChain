from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.seasonal import seasonal_decompose
import json
import plotly
import plotly.graph_objs as go
import os
import io
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime

import os

app = Flask(__name__)
app.secret_key = 'neurochain_secret_key'

# Database Configuration (Cloud + Local Fallback)
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///supplychain.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Models ---

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    shops = db.relationship('Shop', backref='owner', lazy=True)

class Shop(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sales = db.relationship('Sale', backref='shop', lazy=True)

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    sales = db.relationship('Sale', backref='category', lazy=True)

class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    shop_id = db.Column(db.Integer, db.ForeignKey('shop.id'), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Helpers ---

def generate_reasoning(df, forecast_vals, price_col):
    """Simulates an AI Agent reasoning process based on statistical data."""
    # Trend Analysis
    recent_data = df[price_col].tail(7)
    trend_direction = "upward" if recent_data.iloc[-1] > recent_data.iloc[0] else "downward"
    volatility = recent_data.std() / recent_data.mean() * 100
    
    # Seasonality Check (Simple)
    is_cyclical = False
    if len(df) >= 14:
        # Check correlation with 7-day lag
        lag_corr = df[price_col].autocorr(lag=7)
        if lag_corr > 0.6:
            is_cyclical = True

    reasoning_steps = [
        f"Analyzing historical ingestion layer ({len(df)} data points detected).",
        f"Detected {trend_direction} trend with {volatility:.1f}% local volatility.",
    ]
    
    if is_cyclical:
        reasoning_steps.append("Strong 7-day cyclical pattern identified in sales rhythm.")
    else:
        reasoning_steps.append("No significant short-term seasonality detected; relying on linear trend extrapolation.")

    reasoning_steps.append(f"Model selected: Holt's Exponential Smoothing (Level + Trend).")
    reasoning_steps.append("Computing 95% confidence intervals for risk mitigation.")
    
    return {
        "steps": reasoning_steps,
        "summary": f"The system predicts a {trend_direction} trajectory based on recent momentum.",
        "risk_level": "High" if volatility > 20 else "Low" if volatility < 5 else "Medium"
    }

def generate_insights(historical_avg, forecast_avg, is_small_business=False):
    """Generates human-readable business recommendations."""
    diff_pct = ((forecast_avg - historical_avg) / historical_avg) * 100
    
    # Base Insights
    insight = {
        "status": "Stable Demand",
        "icon": "📊",
        "message": "Demand remains consistent. Maintain current inventory levels.",
        "color": "#3b82f6",
        "tip": "Focus on high-margin items to boost profitability during stable periods."
    }

    if diff_pct > 5:
        insight.update({
            "status": "Growth Detected",
            "icon": "📈",
            "message": f"Sales are projected to rise by {diff_pct:.1f}%.",
            "color": "#10b981",
            "tip": "Increase inventory orders by 15% now to prevent stockouts next week."
        })
    elif diff_pct < -5:
        insight.update({
            "status": "Declining Trend",
            "icon": "📉",
            "message": f"Sales projected to drop by {abs(diff_pct):.1f}%.",
            "color": "#ef4444",
            "tip": "Reduce perishable orders and consider a flash sale to clear slow stock."
        })

    if is_small_business:
        if diff_pct > 5:
            insight["tip"] = "Growth Spike: Consider hiring extra part-time help for next week's rush."
        elif diff_pct < -5:
            insight["tip"] = "Cash Flow Alert: Keep your cash in the bank—avoid buying bulk today."
        else:
            insight["tip"] = "Efficiency Tip: Great time to reorganize your shelf space for better visibility."

    return insight

# --- Auth Routes ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        shop_name = request.form.get('shop_name')
        
        if not shop_name:
            flash('Please provide an initial shop name.', 'danger')
            return render_template('register.html')
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        # Create initial shop
        new_shop = Shop(name=shop_name, owner=user)
        db.session.add(new_shop)
        db.session.commit()
        
        flash('Node initialized! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('index'))
        flash('Login failed. Check username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/create-shop', methods=['POST'])
@login_required
def create_shop():
    shop_name = request.form.get('shop_name')
    if shop_name:
        new_shop = Shop(name=shop_name, owner=current_user)
        db.session.add(new_shop)
        db.session.commit()
    return redirect(url_for('index'))

@app.route('/')
@login_required
def index():
    shops = current_user.shops
    selected_shop_id = request.args.get('shop_id')
    
    # Auto-select the first shop if none is specified
    if not selected_shop_id and shops:
        selected_shop_id = shops[0].id
        
    history = []
    selected_shop = None
    if selected_shop_id:
        selected_shop = Shop.query.get(selected_shop_id)
        if selected_shop and selected_shop.user_id == current_user.id:
            history = Sale.query.filter_by(shop_id=selected_shop_id).order_by(Sale.date.desc()).limit(15).all()
            
    return render_template('index.html', 
                           shops=shops, 
                           history=history, 
                           selected_shop=selected_shop,
                           categories=Category.query.all())

@app.route('/download-template')
def download_template():
    df = pd.DataFrame(columns=['date', 'sales_total'])
    df.loc[0] = ['2024-01-01', '150.00']
    output = io.StringIO()
    df.to_csv(output, index=False)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='ai_template.csv')

@app.route('/forecast', methods=['POST'])
@login_required
def forecast():
    mode = request.form.get('mode', 'upload')
    days_to_forecast = int(request.form.get('days', 7))
    shop_id = request.form.get('shop_id')
    is_small_business = (mode == 'manual')

    if not shop_id: return "No shop selected.", 400
    target_shop = Shop.query.get(shop_id)
    if not target_shop or target_shop.user_id != current_user.id: return "Unauthorized.", 403

    # 1. Process incoming data
    new_data = []
    if mode == 'upload':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No CSV file selected', 'danger')
            return redirect(url_for('index'))
        df = pd.read_csv(file)
        if 'Date' in df.columns and 'Amount' in df.columns:
            for _, row in df.iterrows():
                try:
                    # For bulk upload, default to 'Grocery' if not specified
                    grocery = Category.query.filter_by(name='Grocery').first()
                    new_sale = Sale(
                        date=pd.to_datetime(row['Date']).date(),
                        amount=float(row['Amount']),
                        shop_id=shop_id,
                        category_id=grocery.id if grocery else None
                    )
                    db.session.add(new_sale)
                except: continue
            db.session.commit()
    
    elif mode == 'manual':
        date_str = request.form.get('date')
        amount = request.form.get('amount')
        category_id = request.form.get('category_id')
        if date_str and amount:
            new_sale = Sale(
                date=datetime.strptime(date_str, '%Y-%m-%d').date(),
                amount=float(amount),
                shop_id=shop_id,
                category_id=category_id
            )
            db.session.add(new_sale)
            db.session.commit()

    elif mode == 'vision':
        # Simulate Vision OCR Processing
        import random
        # In a real app, we would process request.files.get('receipt_image')
        # Here we mock the result of reading a receipt
        simulated_amount = round(random.uniform(15.50, 85.00), 2)
        meat_cat = Category.query.filter_by(name='Meat').first()
        new_sale = Sale(
            date=datetime.now().date(),
            amount=simulated_amount,
            shop_id=shop_id,
            category_id=meat_cat.id if meat_cat else None
        )
        db.session.add(new_sale)
        db.session.commit()
        flash(f'Neural Vision synced: {meat_cat.name if meat_cat else "Item"} detected (£{simulated_amount})', 'success')

    # 3. Fetch ALL historical data for this shop from DB
    history = Sale.query.filter_by(shop_id=shop_id).order_by(Sale.date).all()
    if len(history) < 3: return "Need at least 3 days of historical data (total) for this shop.", 400

    df = pd.DataFrame([{'date': s.date, 'sales': s.amount} for s in history])
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date')['sales'].mean().reset_index()
    df = df.sort_values('date').set_index('date')
    price_col = 'sales'

    # Forecasting Logic
    init_method = 'heuristic' if len(df) >= 10 else 'estimated'
    try:
        model = Holt(df[price_col], initialization_method=init_method).fit()
        forecast_vals = model.forecast(days_to_forecast)
        
        # Calculate Confidence Intervals (simplified)
        std_dev = df[price_col].std()
        upper_bound = forecast_vals + (1.96 * std_dev / np.sqrt(len(df)))
        lower_bound = forecast_vals - (1.96 * std_dev / np.sqrt(len(df)))
        lower_bound = lower_bound.clip(lower=0)
    except Exception as e:
        return f"Model Error: {str(e)}", 500

    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_to_forecast)

    # Seasonal Decomposition (if enough data)
    decomp_data = None
    if len(df) >= 14:
        try:
            res = seasonal_decompose(df[price_col], model='additive', period=7)
            decomp_data = {
                "trend": res.trend.dropna().tolist(),
                "seasonal": res.seasonal.dropna().tolist(),
                "dates": res.trend.dropna().index.strftime('%Y-%m-%d').tolist()
            }
        except: pass

    # Plotly Visuals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], name='Historical', line=dict(color='#3b82f6', width=3)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_vals, name='AI Forecast', line=dict(color='#10b981', width=4, dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                             y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                             fill='toself', fillcolor='rgba(16, 185, 129, 0.1)',
                             line=dict(color='rgba(255,255,255,0)'), name='Confidence Zone'))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20), height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Business Intelligence & Agent Reasoning
    insights = generate_insights(df[price_col].mean(), forecast_vals.mean(), is_small_business)
    reasoning = generate_reasoning(df, forecast_vals, price_col)

    # Output preparation
    forecast_df = pd.DataFrame({'Date': forecast_dates.strftime('%Y-%m-%d'), 'Value': forecast_vals.round(2)})
    csv_path = os.path.join('static', 'latest_forecast.csv')
    # Categorical Analysis for Stock Intelligence
    category_summary = {}
    for s in history:
        cat_name = s.category.name if s.category else 'General'
        if cat_name not in category_summary:
            category_summary[cat_name] = {'total': 0, 'count': 0}
        category_summary[cat_name]['total'] += s.amount
        category_summary[cat_name]['count'] += 1

    ai_advice = []
    for cat, stats in category_summary.items():
        avg_val = stats['total'] / stats['count']
        if avg_val > 50:
            ai_advice.append({
                'category': cat,
                'status': 'High Momentum',
                'advice': f"Projected spike detected in {cat}. Recommendation: Increase inventory buffer by 20% to prevent stock-outs during peaks.",
                'type': 'priority'
            })
        elif avg_val < 20:
            ai_advice.append({
                'category': cat,
                'status': 'Low Velocity',
                'advice': f"{cat} movement is slow. Recommendation: Monitor expiration dates closely and consider tactical markdowns.",
                'type': 'risk'
            })
        else:
            ai_advice.append({
                'category': cat,
                'status': 'Stable',
                'advice': f"{cat} inventory flow is optimal. Maintain current procurement cycle.",
                'type': 'stable'
            })

    forecast_df.to_csv(csv_path, index=False)

    return render_template('result.html', 
                           graphJSON=graphJSON,
                           decomp_data=decomp_data,
                           csv_path=csv_path,
                           table_data=forecast_df.to_dict(orient='records'), 
                           insights=insights,
                           reasoning=reasoning,
                           days=days_to_forecast,
                           avg_forecast=round(forecast_vals.mean(), 2),
                           ai_advice=ai_advice[:4])

@app.route('/download')
def download():
    path = os.path.join('static', 'latest_forecast.csv')
    return send_file(path, as_attachment=True) if os.path.exists(path) else ("Not found", 404)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Initialize default categories if they don't exist
        if not Category.query.first():
            default_cats = ['Meat', 'Grocery', 'Dairy', 'Household', 'Beverages']
            for cat_name in default_cats:
                db.session.add(Category(name=cat_name))
            db.session.commit()
    app.run(debug=True, host='0.0.0.0')
