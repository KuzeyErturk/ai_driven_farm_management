"""
Risk Analyzer Module
====================
Calculates agricultural risk scores based on weather conditions,
climate change projections, and financial impact.
"""


# Base yields (historical averages in t/ha)
BASE_YIELDS = {
    'Wheat': 8.0,
    'Winter_Barley': 6.5,
    'Spring_Barley': 5.5,
    'Oats': 5.8,
    'OSR': 3.2
}

# Average UK crop prices (£/tonne)
CROP_PRICES = {
    'Wheat': 200,
    'Winter_Barley': 180,
    'Spring_Barley': 190,
    'Oats': 175,
    'OSR': 380
}

# Crop-specific risk sensitivities (higher = more sensitive)
CROP_SENSITIVITIES = {
    'Wheat': {'drought': 0.8, 'frost': 0.6, 'heat': 0.7, 'disease': 0.7, 'flood': 0.8},
    'Winter_Barley': {'drought': 0.7, 'frost': 0.9, 'heat': 0.6, 'disease': 0.8, 'flood': 0.7},
    'Spring_Barley': {'drought': 0.9, 'frost': 1.0, 'heat': 0.8, 'disease': 0.6, 'flood': 0.8},
    'Oats': {'drought': 0.6, 'frost': 0.7, 'heat': 0.9, 'disease': 0.5, 'flood': 0.7},
    'OSR': {'drought': 0.7, 'frost': 0.8, 'heat': 0.5, 'disease': 0.9, 'flood': 0.6}
}


class RiskAnalyzer:
    """Analyzes agricultural risks based on weather and climate data."""

    # Risk thresholds based on historical data
    RISK_THRESHOLDS = {
        'drought': {'Summer_Rain': 150, 'Summer_Tmax': 23},
        'flood': {'Summer_Rain': 300, 'Spring_Rain': 250},
        'frost': {'Spring_Frost': 12, 'Winter_Frost': 40},
        'heat_stress': {'Summer_Tmax': 24},
        'disease': {'Summer_Rain': 280, 'Summer_Tmax': 22}
    }

    # Climate change projections (% change per decade)
    CLIMATE_TRENDS = {
        'Summer_Tmax': 0.8,       # Rising temperatures (°C per decade)
        'Summer_Rain': -5,        # Drier summers (%)
        'Winter_Rain': 8,         # Wetter winters (%)
        'Spring_Frost': -2,       # Fewer frost days
        'Winter_Frost': -3,       # Fewer frost days
        'Extreme_Events': 12      # More volatility (%)
    }

    def calculate_risk_score(self, weather_data, crop):
        """
        Calculate overall risk score (0-100) and individual risk factors.

        Args:
            weather_data: dict with weather parameters
            crop: string crop name

        Returns:
            dict with overall_score, risk_level, individual_risks, recommendations
        """
        risks = []
        sensitivities = CROP_SENSITIVITIES.get(crop, CROP_SENSITIVITIES['Wheat'])

        # Drought risk
        drought_score = self._drought_risk(weather_data)
        if drought_score > 20:
            risks.append(('Drought', round(drought_score * sensitivities['drought'])))

        # Flood risk
        flood_score = self._flood_risk(weather_data)
        if flood_score > 20:
            risks.append(('Flooding', round(flood_score * sensitivities['flood'])))

        # Frost risk
        frost_score = self._frost_risk(weather_data, crop)
        if frost_score > 20:
            risks.append(('Frost Damage', round(frost_score * sensitivities['frost'])))

        # Heat stress
        heat_score = self._heat_risk(weather_data, crop)
        if heat_score > 20:
            risks.append(('Heat Stress', round(heat_score * sensitivities['heat'])))

        # Disease pressure
        disease_score = self._disease_risk(weather_data)
        if disease_score > 20:
            risks.append(('Disease Pressure', round(disease_score * sensitivities['disease'])))

        # Calculate overall score (weighted average with emphasis on highest risks)
        overall = self._calculate_overall(risks)

        return {
            'overall_score': overall,
            'risk_level': self._score_to_level(overall),
            'individual_risks': risks,
            'recommendations': self._get_recommendations(risks, crop)
        }

    def _drought_risk(self, weather):
        """Calculate drought risk based on rainfall and temperature."""
        summer_rain = weather.get('Summer_Rain', 200)
        summer_tmax = weather.get('Summer_Tmax', 21)
        spring_rain = weather.get('Spring_Rain', 175)

        score = 0

        # Low summer rainfall increases drought risk
        if summer_rain < 180:
            score += (180 - summer_rain) / 80 * 40  # Up to 40 points

        # Low spring rainfall compounds the issue
        if spring_rain < 150:
            score += (150 - spring_rain) / 50 * 20  # Up to 20 points

        # High temperatures increase evapotranspiration
        if summer_tmax > 22:
            score += (summer_tmax - 22) / 4 * 40  # Up to 40 points

        return min(score, 100)

    def _flood_risk(self, weather):
        """Calculate flood risk based on excessive rainfall."""
        summer_rain = weather.get('Summer_Rain', 200)
        spring_rain = weather.get('Spring_Rain', 175)

        score = 0

        # Excessive summer rain
        if summer_rain > 280:
            score += (summer_rain - 280) / 120 * 50  # Up to 50 points

        # Excessive spring rain
        if spring_rain > 220:
            score += (spring_rain - 220) / 80 * 50  # Up to 50 points

        return min(score, 100)

    def _frost_risk(self, weather, crop):
        """Calculate frost risk based on frost days."""
        spring_frost = weather.get('Spring_Frost', 8)
        winter_frost = weather.get('Winter_Frost', 25)

        score = 0

        # Spring frost is most damaging
        if spring_frost > 8:
            score += (spring_frost - 8) / 12 * 60  # Up to 60 points

        # Winter frost matters for winter crops
        if crop in ['Wheat', 'Winter_Barley', 'OSR']:
            if winter_frost > 35:
                score += (winter_frost - 35) / 15 * 40  # Up to 40 points

        return min(score, 100)

    def _heat_risk(self, weather, crop):
        """Calculate heat stress risk."""
        summer_tmax = weather.get('Summer_Tmax', 21)
        summer_sun = weather.get('Summer_Sun', 550)

        score = 0

        # High temperatures cause heat stress
        if summer_tmax > 23:
            score += (summer_tmax - 23) / 3 * 70  # Up to 70 points

        # Excessive sun compounds heat stress (less cloud cover)
        if summer_sun > 620 and summer_tmax > 22:
            score += (summer_sun - 620) / 80 * 30  # Up to 30 points

        return min(score, 100)

    def _disease_risk(self, weather):
        """Calculate disease pressure based on humidity conditions."""
        summer_rain = weather.get('Summer_Rain', 200)
        summer_tmax = weather.get('Summer_Tmax', 21)
        grain_filling_rain = weather.get('Grain_Filling_Rain', 150)

        score = 0

        # High moisture promotes fungal diseases
        if summer_rain > 250:
            score += (summer_rain - 250) / 150 * 40  # Up to 40 points

        # Moderate temperatures with moisture = disease pressure
        if 18 <= summer_tmax <= 22 and summer_rain > 220:
            score += 30  # Perfect disease conditions

        # Wet grain filling period increases disease risk
        if grain_filling_rain > 180:
            score += (grain_filling_rain - 180) / 70 * 30  # Up to 30 points

        return min(score, 100)

    def _calculate_overall(self, risks):
        """Calculate overall risk score from individual risks."""
        if not risks:
            return 15  # Base risk level (nothing is ever risk-free)

        # Weight highest risks more heavily
        scores = sorted([score for _, score in risks], reverse=True)

        if len(scores) == 1:
            return scores[0]

        # Weighted: highest risk counts most, others add diminishing amounts
        overall = scores[0]
        for i, score in enumerate(scores[1:], start=1):
            overall += score * (0.5 ** i)  # Diminishing weights

        return min(round(overall), 100)

    def _score_to_level(self, score):
        """Convert numeric score to risk level."""
        if score < 25:
            return 'Low'
        elif score < 50:
            return 'Moderate'
        elif score < 75:
            return 'High'
        else:
            return 'Critical'

    def project_climate_impact(self, crop, years_ahead=10):
        """
        Project yield changes due to climate trends.

        Args:
            crop: string crop name
            years_ahead: years into the future (default 10)

        Returns:
            dict with current/projected yields and key factors
        """
        base_yield = BASE_YIELDS.get(crop, 7.0)
        decades = years_ahead / 10

        # Calculate projected weather shifts
        projected_temp_increase = self.CLIMATE_TRENDS['Summer_Tmax'] * decades
        projected_rain_change = self.CLIMATE_TRENDS['Summer_Rain'] * decades
        projected_extreme_increase = self.CLIMATE_TRENDS['Extreme_Events'] * decades

        # Estimate yield impact
        # Temperature impact: 3% loss per degree above optimal
        temp_impact = -0.03 * projected_temp_increase

        # Rainfall variability impact
        rain_impact = -0.01 * abs(projected_rain_change)

        # Extreme event impact (increased volatility)
        extreme_impact = -0.002 * projected_extreme_increase

        total_impact = temp_impact + rain_impact + extreme_impact
        projected_yield = base_yield * (1 + total_impact)

        # Calculate adaptation benefit if measures taken
        adapted_yield = projected_yield * 1.05  # 5% recovery with adaptation

        return {
            'current_expected': base_yield,
            'projected_yield': round(projected_yield, 2),
            'adapted_yield': round(adapted_yield, 2),
            'change_percent': round(total_impact * 100, 1),
            'years_ahead': years_ahead,
            'key_factors': [
                {
                    'factor': 'Temperature',
                    'change': f'+{projected_temp_increase:.1f}°C',
                    'impact': 'Negative'
                },
                {
                    'factor': 'Summer Rainfall',
                    'change': f'{projected_rain_change:+.0f}%',
                    'impact': 'Negative'
                },
                {
                    'factor': 'Extreme Events',
                    'change': f'+{projected_extreme_increase:.0f}%',
                    'impact': 'Negative'
                },
                {
                    'factor': 'Frost Days',
                    'change': f'{self.CLIMATE_TRENDS["Spring_Frost"] * decades:+.0f} days',
                    'impact': 'Positive'
                }
            ]
        }

    def calculate_financial_risk(self, crop, region, weather_data, hectares=100):
        """
        Calculate potential financial impact of risks.

        Args:
            crop: string crop name
            region: string region name
            weather_data: dict with weather parameters
            hectares: farm size in hectares

        Returns:
            dict with expected income, income at risk, worst case
        """
        base_yield = BASE_YIELDS.get(crop, 7.0)
        price = CROP_PRICES.get(crop, 200)

        # Get risk assessment
        risk_data = self.calculate_risk_score(weather_data, crop)

        # Estimate yield reduction based on risk score
        # At 100 risk score, expect up to 40% yield loss
        risk_score = risk_data['overall_score']
        yield_reduction = (risk_score / 100) * 0.4

        expected_yield = base_yield * (1 - yield_reduction)

        # Calculate financial figures
        expected_income = base_yield * price * hectares
        at_risk_income = (base_yield - expected_yield) * price * hectares
        worst_case_loss = base_yield * 0.5 * price * hectares  # 50% crop failure

        # Insurance recommendation threshold
        insurance_recommended = risk_score > 45

        return {
            'expected_income': round(expected_income),
            'expected_income_formatted': f'£{expected_income:,.0f}',
            'income_at_risk': round(at_risk_income),
            'income_at_risk_formatted': f'£{at_risk_income:,.0f}',
            'worst_case_loss': round(worst_case_loss),
            'worst_case_loss_formatted': f'£{worst_case_loss:,.0f}',
            'expected_yield': round(expected_yield, 2),
            'yield_reduction_percent': round(yield_reduction * 100, 1),
            'price_per_tonne': price,
            'insurance_recommended': insurance_recommended,
            'hectares': hectares
        }

    def _get_recommendations(self, risks, crop):
        """Generate actionable recommendations based on identified risks."""
        recommendations = []

        for risk_type, score in risks:
            if risk_type == 'Drought' and score > 30:
                recommendations.append({
                    'priority': 'High' if score > 50 else 'Medium',
                    'risk_type': 'Drought',
                    'action': 'Implement irrigation scheduling',
                    'detail': 'Consider soil moisture monitoring and deficit irrigation techniques',
                    'timing': 'Immediate'
                })
            elif risk_type == 'Flooding' and score > 30:
                recommendations.append({
                    'priority': 'High' if score > 50 else 'Medium',
                    'risk_type': 'Flooding',
                    'action': 'Improve field drainage',
                    'detail': 'Check drains are clear, consider installing additional drainage if recurring issue',
                    'timing': 'Before next growing season'
                })
            elif risk_type == 'Frost Damage' and score > 30:
                recommendations.append({
                    'priority': 'High' if score > 50 else 'Medium',
                    'risk_type': 'Frost',
                    'action': 'Adjust planting schedule',
                    'detail': 'Consider later spring planting dates or cold-tolerant varieties',
                    'timing': 'Planning stage'
                })
            elif risk_type == 'Heat Stress' and score > 30:
                recommendations.append({
                    'priority': 'High' if score > 50 else 'Medium',
                    'risk_type': 'Heat',
                    'action': 'Select heat-tolerant varieties',
                    'detail': 'Consider earlier maturing varieties to avoid peak summer heat',
                    'timing': 'Variety selection'
                })
            elif risk_type == 'Disease Pressure' and score > 30:
                recommendations.append({
                    'priority': 'High' if score > 50 else 'Medium',
                    'risk_type': 'Disease',
                    'action': 'Implement preventive fungicide program',
                    'detail': 'Regular field scouting and timely application at T1/T2 stages',
                    'timing': 'Growing season'
                })

        # Always add long-term adaptation recommendation
        recommendations.append({
            'priority': 'Long-term',
            'risk_type': 'Climate',
            'action': 'Diversify crop portfolio',
            'detail': 'Spread risk across multiple crop types and consider climate-resilient varieties',
            'timing': 'Strategic planning'
        })

        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Long-term': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return recommendations

    def get_historical_risk_trend(self, crop, years=10):
        """
        Generate historical risk trend data for visualization.

        Args:
            crop: string crop name
            years: number of years of history

        Returns:
            list of dicts with year and risk scores
        """
        import random
        random.seed(42)  # Reproducible demo data

        trend = []
        base_risk = 35

        for i in range(years):
            year = 2015 + i
            # Simulate increasing risk over time due to climate change
            climate_increase = i * 1.5
            # Add some random variation
            variation = random.uniform(-8, 8)

            risk_score = min(base_risk + climate_increase + variation, 85)
            trend.append({
                'year': year,
                'risk_score': round(risk_score),
                'risk_level': self._score_to_level(risk_score)
            })

        return trend
