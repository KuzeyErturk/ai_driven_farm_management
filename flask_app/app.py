"""
Agricultural AI Flask Application
==================================
Web interface for crop yield prediction and scenario analysis.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json

from utils.risk_analyzer import RiskAnalyzer

app = Flask(__name__)

# Load configuration
CROPS = ['Wheat', 'Winter_Barley', 'Spring_Barley', 'Oats', 'OSR']
REGIONS = ['England', 'Wales', 'Scotland', 'Northern Ireland']

# Weather feature ranges (based on historical data)
WEATHER_RANGES = {
    'Summer_Rain': {'min': 100, 'max': 400, 'default': 200, 'unit': 'mm'},
    'Summer_Sun': {'min': 400, 'max': 700, 'default': 550, 'unit': 'hours'},
    'Winter_Frost': {'min': 10, 'max': 50, 'default': 25, 'unit': 'days'},
    'Spring_Rain': {'min': 100, 'max': 300, 'default': 175, 'unit': 'mm'},
    'Spring_Frost': {'min': 0, 'max': 20, 'default': 8, 'unit': 'days'},
    'Summer_Tmax': {'min': 18, 'max': 26, 'default': 21, 'unit': '°C'},
    'Grain_Filling_Rain': {'min': 50, 'max': 250, 'default': 150, 'unit': 'mm'},
    'Grain_Filling_Sun': {'min': 200, 'max': 500, 'default': 350, 'unit': 'hours'},
}

# Historical yield data for visualization
def load_historical_data():
    """Load historical yield data for trends."""
    try:
        df = pd.read_csv('../data/processed/regional_crop_yield_weather_2004_2024.csv')
        return df
    except FileNotFoundError:
        # Return mock data if file not found
        return pd.DataFrame({
            'Year': list(range(2004, 2025)) * 4,
            'Crop': ['Wheat'] * 21 + ['Barley'] * 21 + ['Oats'] * 21 + ['OSR'] * 21,
            'Yield_t_per_ha': np.random.uniform(5, 9, 84)
        })


@app.route('/')
def index():
    """Main page with prediction form."""
    return render_template('index.html',
                           crops=CROPS,
                           regions=REGIONS,
                           weather_ranges=WEATHER_RANGES)


@app.route('/predict', methods=['POST'])
def predict():
    """Make yield prediction based on weather inputs."""
    try:
        data = request.get_json()
        crop = data.get('crop', 'Wheat')
        region = data.get('region', 'England')

        # Get weather inputs
        weather = {
            'Summer_Rain': float(data.get('summer_rain', 200)),
            'Summer_Sun': float(data.get('summer_sun', 550)),
            'Winter_Frost': float(data.get('winter_frost', 25)),
            'Spring_Rain': float(data.get('spring_rain', 175)),
            'Spring_Frost': float(data.get('spring_frost', 8)),
            'Summer_Tmax': float(data.get('summer_tmax', 21)),
            'Grain_Filling_Rain': float(data.get('grain_filling_rain', 150)),
            'Grain_Filling_Sun': float(data.get('grain_filling_sun', 350)),
        }

        # Load model (or use fallback prediction)
        model_path = f'models/{crop.lower()}_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            features = np.array([[weather[f] for f in WEATHER_RANGES.keys()]])
            prediction = model.predict(features)[0]
        else:
            # Fallback: use simple rule-based prediction
            prediction = calculate_fallback_prediction(crop, weather)

        # Calculate confidence based on how extreme the weather is
        confidence = calculate_confidence(weather)

        # Determine if conditions are extreme
        extremes = detect_extreme_conditions(weather)

        # Get historical average for comparison
        historical_avg = get_historical_average(crop)

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'confidence': confidence,
            'unit': 't/ha',
            'crop': crop,
            'region': region,
            'historical_avg': historical_avg,
            'deviation': round(prediction - historical_avg, 2),
            'extreme_conditions': extremes
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def calculate_fallback_prediction(crop, weather):
    """Simple rule-based prediction when model not available."""
    # Base yields (historical averages)
    base_yields = {
        'Wheat': 8.0,
        'Winter_Barley': 6.5,
        'Spring_Barley': 5.5,
        'Oats': 5.8,
        'OSR': 3.2
    }

    # Optimal weather conditions for maximum yield
    optimal = {
        'Summer_Rain': 200,      # mm - optimal range
        'Summer_Sun': 580,       # hours
        'Winter_Frost': 20,      # days
        'Spring_Frost': 5,       # days
        'Spring_Rain': 175,      # mm
        'Summer_Tmax': 21,       # °C
        'Grain_Filling_Rain': 140,  # mm
        'Grain_Filling_Sun': 380    # hours
    }

    base = base_yields.get(crop, 7.0)
    adjustment = 0.0

    # Summer rain: too much or too little is bad (optimal around 200mm)
    rain_diff = abs(weather['Summer_Rain'] - optimal['Summer_Rain'])
    adjustment -= (rain_diff / 100) * 0.4  # Up to -0.8 for extremes

    # Summer sunshine: more is generally better, but diminishing returns
    sun_diff = weather['Summer_Sun'] - optimal['Summer_Sun']
    if sun_diff > 0:
        adjustment += min(sun_diff / 120, 0.4)  # Bonus for more sun
    else:
        adjustment += (sun_diff / 130) * 0.5  # Penalty for less sun

    # Winter frost: more frost days = worse
    frost_excess = max(0, weather['Winter_Frost'] - optimal['Winter_Frost'])
    adjustment -= (frost_excess / 30) * 0.5

    # Spring frost: very damaging
    spring_frost_excess = max(0, weather['Spring_Frost'] - optimal['Spring_Frost'])
    adjustment -= (spring_frost_excess / 15) * 0.8

    # Spring rain: moderate is good
    spring_rain_diff = abs(weather['Spring_Rain'] - optimal['Spring_Rain'])
    adjustment -= (spring_rain_diff / 125) * 0.3

    # Summer max temperature: heat stress above 22°C
    if weather['Summer_Tmax'] > 22:
        adjustment -= ((weather['Summer_Tmax'] - 22) / 4) * 0.6
    elif weather['Summer_Tmax'] < 19:
        adjustment -= ((19 - weather['Summer_Tmax']) / 3) * 0.3

    # Grain filling conditions
    gf_rain_diff = abs(weather['Grain_Filling_Rain'] - optimal['Grain_Filling_Rain'])
    adjustment -= (gf_rain_diff / 110) * 0.3

    gf_sun_diff = weather['Grain_Filling_Sun'] - optimal['Grain_Filling_Sun']
    if gf_sun_diff > 0:
        adjustment += min(gf_sun_diff / 120, 0.3)
    else:
        adjustment += (gf_sun_diff / 150) * 0.4

    prediction = base + adjustment
    return round(max(prediction, 2.0), 2)  # Minimum yield of 2.0


def calculate_confidence(weather):
    """Calculate prediction confidence based on weather extremity."""
    extreme_count = 0

    if weather['Summer_Rain'] > 350 or weather['Summer_Rain'] < 120:
        extreme_count += 1
    if weather['Winter_Frost'] > 45:
        extreme_count += 1
    if weather['Spring_Frost'] > 18:
        extreme_count += 1
    if weather['Summer_Tmax'] > 25:
        extreme_count += 1
    if weather['Summer_Sun'] < 420:
        extreme_count += 1

    if extreme_count == 0:
        return 'High'
    elif extreme_count <= 2:
        return 'Medium'
    else:
        return 'Low'


def detect_extreme_conditions(weather):
    """Detect which weather conditions are extreme and provide recommendations."""
    extremes = []

    if weather['Summer_Rain'] > 320:
        extremes.append({
            'condition': 'High Summer Rain',
            'impact': 'Negative',
            'description': 'Excess rain during grain filling',
            'recommendation': 'Monitor crops for signs of disease. Consider preventive fungicide application. Check field drainage is functioning properly.'
        })
    if weather['Summer_Rain'] < 130:
        extremes.append({
            'condition': 'Low Summer Rain',
            'impact': 'Negative',
            'description': 'Drought stress during grain filling',
            'recommendation': 'Implement irrigation if available. Consider drought-tolerant varieties for next season. Monitor soil moisture levels closely.'
        })
    if weather['Winter_Frost'] > 40:
        extremes.append({
            'condition': 'Severe Winter',
            'impact': 'Negative',
            'description': 'Frost damage to overwintering crops',
            'recommendation': 'Assess crop damage in early spring. Consider reseeding if necessary. Plan for frost-tolerant varieties next season.'
        })
    if weather['Spring_Frost'] > 15:
        extremes.append({
            'condition': 'Late Spring Frost',
            'impact': 'Negative',
            'description': 'Damage to flowering/establishment',
            'recommendation': 'Delay planting if possible. Protect young crops with covers. Select cold-tolerant varieties for future plantings.'
        })
    if weather['Summer_Tmax'] > 24:
        extremes.append({
            'condition': 'Heat Stress',
            'impact': 'Negative',
            'description': 'Accelerated senescence',
            'recommendation': 'Irrigate during heat waves to reduce stress. Harvest early if grain is mature. Plan for earlier-maturing varieties next year.'
        })
    if weather['Summer_Sun'] > 650:
        extremes.append({
            'condition': 'High Sunshine',
            'impact': 'Positive',
            'description': 'Excellent photosynthesis conditions',
            'recommendation': 'Monitor for moisture stress despite good conditions. Optimize harvest timing to capture quality benefits.'
        })

    return extremes


def get_historical_average(crop):
    """Get historical average yield for crop."""
    averages = {
        'Wheat': 8.0,
        'Winter_Barley': 6.5,
        'Spring_Barley': 5.5,
        'Oats': 5.8,
        'OSR': 3.2
    }
    return averages.get(crop, 6.5)


@app.route('/scenario')
def scenario():
    """Scenario analysis page with interactive sliders."""
    return render_template('scenario.html',
                           crops=CROPS,
                           weather_ranges=WEATHER_RANGES)


@app.route('/risk')
def risk():
    """Risk assessment dashboard page."""
    return render_template('risk.html',
                           crops=CROPS,
                           regions=REGIONS,
                           weather_ranges=WEATHER_RANGES)


@app.route('/api/risk', methods=['POST'])
def calculate_risk():
    """API endpoint for risk assessment calculation."""
    try:
        data = request.get_json()
        crop = data.get('crop', 'Wheat')
        region = data.get('region', 'England')
        hectares = float(data.get('hectares', 100))
        years_ahead = int(data.get('years_ahead', 10))

        # Get weather inputs
        weather = {
            'Summer_Rain': float(data.get('Summer_Rain', 200)),
            'Summer_Sun': float(data.get('Summer_Sun', 550)),
            'Winter_Frost': float(data.get('Winter_Frost', 25)),
            'Spring_Rain': float(data.get('Spring_Rain', 175)),
            'Spring_Frost': float(data.get('Spring_Frost', 8)),
            'Summer_Tmax': float(data.get('Summer_Tmax', 21)),
            'Grain_Filling_Rain': float(data.get('Grain_Filling_Rain', 150)),
            'Grain_Filling_Sun': float(data.get('Grain_Filling_Sun', 350)),
        }

        analyzer = RiskAnalyzer()

        # Calculate all risk components
        risk_assessment = analyzer.calculate_risk_score(weather, crop)
        climate_projection = analyzer.project_climate_impact(crop, years_ahead)
        financial_risk = analyzer.calculate_financial_risk(crop, region, weather, hectares)
        historical_trend = analyzer.get_historical_risk_trend(crop)

        return jsonify({
            'success': True,
            'risk_assessment': risk_assessment,
            'climate_projection': climate_projection,
            'financial_risk': financial_risk,
            'historical_trend': historical_trend
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("Starting Agricultural AI Flask Application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
