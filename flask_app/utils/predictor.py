"""
Predictor Utility
==================
Handles model loading and prediction logic for the Flask application.
"""

import joblib
import numpy as np
import os
import json

# Feature configuration
FEATURES = [
    'Summer_Rain', 'Summer_Sun', 'Winter_Frost', 'Spring_Rain',
    'Spring_Frost', 'Summer_Tmax', 'Grain_Filling_Rain', 'Grain_Filling_Sun'
]

# Historical averages for comparison
HISTORICAL_AVERAGES = {
    'Wheat': 8.0,
    'Winter_Barley': 6.5,
    'Spring_Barley': 5.5,
    'Oats': 5.8,
    'OSR': 3.2
}

# Weather thresholds for extreme detection
EXTREME_THRESHOLDS = {
    'Summer_Rain': {'high': 320, 'low': 130},
    'Winter_Frost': {'high': 40},
    'Spring_Frost': {'high': 15},
    'Summer_Tmax': {'high': 24},
    'Summer_Sun': {'low': 420},
}


class CropPredictor:
    """Handles crop yield predictions."""

    def __init__(self, models_dir=None):
        """Initialize predictor with model directory."""
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '../models')
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self._load_models()

    def _load_models(self):
        """Load all trained models and scalers."""
        for crop in HISTORICAL_AVERAGES.keys():
            model_name = crop.lower().replace(' ', '_')
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            scaler_path = os.path.join(self.models_dir, f'{model_name}_scaler.pkl')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[crop] = joblib.load(model_path)
                self.scalers[crop] = joblib.load(scaler_path)

    def predict(self, crop, weather_data):
        """
        Make a yield prediction for the given crop and weather conditions.

        Args:
            crop: Crop name (e.g., 'Wheat', 'Winter_Barley')
            weather_data: Dictionary of weather features

        Returns:
            Dictionary with prediction, confidence, and extreme conditions
        """
        # Check if model is available
        if crop in self.models:
            prediction = self._model_predict(crop, weather_data)
        else:
            prediction = self._fallback_predict(crop, weather_data)

        # Calculate confidence and detect extremes
        confidence = self._calculate_confidence(weather_data)
        extremes = self._detect_extremes(weather_data)

        # Get historical comparison
        historical_avg = HISTORICAL_AVERAGES.get(crop, 7.0)
        deviation = prediction - historical_avg

        return {
            'prediction': round(prediction, 2),
            'confidence': confidence,
            'historical_avg': historical_avg,
            'deviation': round(deviation, 2),
            'extreme_conditions': extremes
        }

    def _model_predict(self, crop, weather_data):
        """Use trained model for prediction."""
        model = self.models[crop]
        scaler = self.scalers[crop]

        # Extract features in correct order
        features = np.array([[weather_data.get(f, 0) for f in FEATURES]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return prediction

    def _fallback_predict(self, crop, weather_data):
        """Simple rule-based prediction when model not available."""
        base = HISTORICAL_AVERAGES.get(crop, 7.0)

        # Adjust based on weather
        summer_rain = weather_data.get('Summer_Rain', 200)
        summer_sun = weather_data.get('Summer_Sun', 550)
        winter_frost = weather_data.get('Winter_Frost', 25)
        spring_frost = weather_data.get('Spring_Frost', 8)
        summer_tmax = weather_data.get('Summer_Tmax', 21)

        # Summer rain impact (excess is negative)
        if summer_rain > 300:
            base -= 0.8
        elif summer_rain < 150:
            base -= 0.3

        # Sunshine bonus
        if summer_sun > 600:
            base += 0.3
        elif summer_sun < 450:
            base -= 0.4

        # Frost penalties
        if winter_frost > 40:
            base -= 0.5
        if spring_frost > 15:
            base -= 0.6

        # Heat stress
        if summer_tmax > 24:
            base -= 0.4

        return max(base, 2.0)

    def _calculate_confidence(self, weather_data):
        """Calculate prediction confidence based on weather extremity."""
        extreme_count = 0

        summer_rain = weather_data.get('Summer_Rain', 200)
        winter_frost = weather_data.get('Winter_Frost', 25)
        spring_frost = weather_data.get('Spring_Frost', 8)
        summer_tmax = weather_data.get('Summer_Tmax', 21)
        summer_sun = weather_data.get('Summer_Sun', 550)

        if summer_rain > 350 or summer_rain < 120:
            extreme_count += 1
        if winter_frost > 45:
            extreme_count += 1
        if spring_frost > 18:
            extreme_count += 1
        if summer_tmax > 25:
            extreme_count += 1
        if summer_sun < 420:
            extreme_count += 1

        if extreme_count == 0:
            return 'High'
        elif extreme_count <= 2:
            return 'Medium'
        else:
            return 'Low'

    def _detect_extremes(self, weather_data):
        """Detect extreme weather conditions."""
        extremes = []

        summer_rain = weather_data.get('Summer_Rain', 200)
        winter_frost = weather_data.get('Winter_Frost', 25)
        spring_frost = weather_data.get('Spring_Frost', 8)
        summer_tmax = weather_data.get('Summer_Tmax', 21)
        summer_sun = weather_data.get('Summer_Sun', 550)

        if summer_rain > 320:
            extremes.append({
                'condition': 'High Summer Rain',
                'impact': 'Negative',
                'description': 'Excess rain during grain filling reduces quality and yield'
            })
        if summer_rain < 130:
            extremes.append({
                'condition': 'Low Summer Rain',
                'impact': 'Negative',
                'description': 'Drought stress during critical growth period'
            })
        if winter_frost > 40:
            extremes.append({
                'condition': 'Severe Winter',
                'impact': 'Negative',
                'description': 'Extended frost damages overwintering crops'
            })
        if spring_frost > 15:
            extremes.append({
                'condition': 'Late Spring Frost',
                'impact': 'Negative',
                'description': 'Frost damage during flowering and establishment'
            })
        if summer_tmax > 24:
            extremes.append({
                'condition': 'Heat Stress',
                'impact': 'Negative',
                'description': 'High temperatures accelerate senescence'
            })
        if summer_sun > 650:
            extremes.append({
                'condition': 'High Sunshine',
                'impact': 'Positive',
                'description': 'Excellent conditions for photosynthesis'
            })

        return extremes


# Singleton instance
_predictor = None


def get_predictor():
    """Get or create predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CropPredictor()
    return _predictor
