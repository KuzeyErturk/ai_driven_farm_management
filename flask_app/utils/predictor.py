"""
Predictor Utility
==================
Handles model loading and prediction logic for the Flask application.
Uses crop-specific features and algorithms from model_config.json.
"""

import joblib
import numpy as np
import os
import json

# Historical averages for comparison
HISTORICAL_AVERAGES = {
    'Wheat': 8.0,
    'Winter_Barley': 6.5,
    'Spring_Barley': 5.5,
    'Oats': 5.8,
    'OSR': 3.2
}


class CropPredictor:
    """Handles crop yield predictions with crop-specific models and features."""

    def __init__(self, models_dir=None):
        """Initialize predictor with model directory."""
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '../models')
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.config = {}
        self._load_config()
        self._load_models()

    def _load_config(self):
        """Load model_config.json with per-crop feature lists."""
        config_path = os.path.join(self.models_dir, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

    def _load_models(self):
        """Load all trained models and scalers."""
        for crop in self.config.keys():
            model_name = crop.lower()
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            scaler_path = os.path.join(self.models_dir, f'{model_name}_scaler.pkl')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[crop] = joblib.load(model_path)
                self.scalers[crop] = joblib.load(scaler_path)

    def get_features(self, crop):
        """Get the feature list for a crop."""
        crop_config = self.config.get(crop, {})
        return list(crop_config.get('features', {}).keys())

    def predict(self, crop, weather_data):
        """
        Make a yield prediction for the given crop and weather conditions.

        Args:
            crop: Crop name (e.g., 'Wheat', 'Winter_Barley')
            weather_data: Dictionary of weather features

        Returns:
            Dictionary with prediction, confidence, and extreme conditions
        """
        if crop in self.models and crop in self.config:
            prediction = self._model_predict(crop, weather_data)
        else:
            prediction = self._fallback_predict(crop, weather_data)

        confidence = self._calculate_confidence(weather_data)
        extremes = self._detect_extremes(weather_data)

        historical_avg = HISTORICAL_AVERAGES.get(crop, 7.0)
        deviation = prediction - historical_avg

        crop_config = self.config.get(crop, {})

        return {
            'prediction': round(prediction, 2),
            'confidence': confidence,
            'model_type': crop_config.get('model_type', 'Fallback'),
            'historical_avg': historical_avg,
            'deviation': round(deviation, 2),
            'extreme_conditions': extremes
        }

    def _model_predict(self, crop, weather_data):
        """Use trained model for prediction with crop-specific features."""
        model = self.models[crop]
        scaler = self.scalers[crop]
        feature_names = self.get_features(crop)

        # Extract features in correct order, using defaults from config
        crop_features = self.config[crop]['features']
        features = []
        for feat in feature_names:
            default = crop_features[feat].get('default', 0)
            features.append(float(weather_data.get(feat, default)))

        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]

        return prediction

    def _fallback_predict(self, crop, weather_data):
        """Simple rule-based prediction when model not available."""
        base = HISTORICAL_AVERAGES.get(crop, 7.0)

        summer_rain = float(weather_data.get('Summer_Rain', 200))
        summer_sun = float(weather_data.get('Summer_Sun', 550))
        winter_frost = float(weather_data.get('Winter_Frost', 25))

        if summer_rain > 300:
            base -= 0.8
        elif summer_rain < 150:
            base -= 0.3

        if summer_sun > 600:
            base += 0.3
        elif summer_sun < 450:
            base -= 0.4

        if winter_frost > 40:
            base -= 0.5

        return max(base, 2.0)

    def _calculate_confidence(self, weather_data):
        """Calculate prediction confidence based on weather extremity."""
        extreme_count = 0

        summer_rain = float(weather_data.get('Summer_Rain', 200))
        winter_frost = float(weather_data.get('Winter_Frost', 25))
        spring_rain = float(weather_data.get('Spring_Rain', 175))
        summer_sun = float(weather_data.get('Summer_Sun', 550))

        if summer_rain > 350 or summer_rain < 120:
            extreme_count += 1
        if winter_frost > 45:
            extreme_count += 1
        if spring_rain > 280 or spring_rain < 100:
            extreme_count += 1
        if summer_sun < 350:
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

        winter_frost = float(weather_data.get('Winter_Frost', 25))
        spring_rain = float(weather_data.get('Spring_Rain', 175))
        grain_filling_rain = float(weather_data.get('Grain_Filling_Rain', 150))

        if grain_filling_rain > 300:
            extremes.append({
                'condition': 'High Grain Filling Rain',
                'impact': 'Negative',
                'description': 'Excess rain during grain filling reduces quality and yield'
            })
        if winter_frost > 40:
            extremes.append({
                'condition': 'Severe Winter',
                'impact': 'Negative',
                'description': 'Extended frost damages overwintering crops'
            })
        if spring_rain > 350:
            extremes.append({
                'condition': 'Wet Spring',
                'impact': 'Negative',
                'description': 'Waterlogging delays planting and establishment'
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
