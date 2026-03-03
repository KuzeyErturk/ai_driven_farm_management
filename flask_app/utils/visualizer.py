"""
Visualizer Utility
===================
Generates charts and visualizations for the Flask application.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import base64

# Color scheme
COLORS = {
    'primary': '#198754',
    'secondary': '#6c757d',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8'
}

# Historical yield data (simplified for visualization)
HISTORICAL_YIELDS = {
    'Wheat': {
        'years': list(range(2004, 2025)),
        'yields': [7.7, 7.9, 8.0, 7.2, 8.4, 8.0, 7.7, 7.8, 6.7, 7.4, 8.6, 8.8,
                  7.9, 8.0, 7.8, 8.5, 8.4, 7.1, 7.5, 8.1, 8.3]
    },
    'Winter_Barley': {
        'years': list(range(2004, 2025)),
        'yields': [6.3, 6.5, 6.2, 5.9, 6.6, 6.4, 6.2, 6.5, 5.8, 6.0, 7.0, 6.9,
                  6.4, 6.5, 6.3, 6.8, 6.7, 5.8, 6.2, 6.6, 6.8]
    },
    'Spring_Barley': {
        'years': list(range(2004, 2025)),
        'yields': [5.2, 5.4, 5.3, 4.9, 5.6, 5.5, 5.3, 5.6, 4.8, 5.1, 5.9, 5.8,
                  5.4, 5.5, 5.3, 5.7, 5.6, 4.9, 5.2, 5.5, 5.7]
    },
    'Oats': {
        'years': list(range(2004, 2025)),
        'yields': [5.5, 5.7, 5.6, 5.2, 5.9, 5.8, 5.6, 5.8, 5.1, 5.4, 6.2, 6.1,
                  5.7, 5.8, 5.6, 6.0, 5.9, 5.2, 5.5, 5.8, 6.0]
    },
    'OSR': {
        'years': list(range(2004, 2025)),
        'yields': [3.1, 3.3, 3.2, 2.9, 3.4, 3.3, 3.1, 3.4, 2.8, 3.0, 3.5, 3.4,
                  3.1, 3.2, 3.0, 3.3, 3.2, 2.8, 3.0, 3.2, 3.3]
    }
}


def create_yield_trend_chart(crop, output_path=None):
    """
    Create a yield trend chart for a specific crop.

    Args:
        crop: Crop name
        output_path: Optional path to save the chart

    Returns:
        If output_path is None, returns base64 encoded image
    """
    if crop not in HISTORICAL_YIELDS:
        return None

    data = HISTORICAL_YIELDS[crop]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot yield trend
    years = data['years']
    yields = data['yields']

    # Color points by extreme years
    colors = []
    for year in years:
        if year == 2010:  # Cold winter
            colors.append(COLORS['info'])
        elif year == 2012:  # Wet summer
            colors.append(COLORS['danger'])
        else:
            colors.append(COLORS['success'])

    ax.plot(years, yields, color=COLORS['primary'], linewidth=2, alpha=0.7)
    ax.scatter(years, yields, c=colors, s=60, zorder=5, edgecolors='white')

    # Add mean line
    mean_yield = np.mean(yields)
    ax.axhline(y=mean_yield, color=COLORS['secondary'], linestyle='--',
               alpha=0.5, label=f'Mean: {mean_yield:.2f} t/ha')

    # Annotate extreme years
    for i, year in enumerate(years):
        if year == 2010:
            ax.annotate('Cold\nWinter', (year, yields[i]), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=8, color=COLORS['info'])
        elif year == 2012:
            ax.annotate('Wet\nSummer', (year, yields[i]), textcoords="offset points",
                       xytext=(0, -25), ha='center', fontsize=8, color=COLORS['danger'])

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Yield (t/ha)', fontsize=12)
    ax.set_title(f'{crop.replace("_", " ")} - Historical Yield Trend (2004-2024)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2003, 2025)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        # Return base64 encoded image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_feature_importance_chart(crop, features, importances, output_path=None):
    """
    Create a horizontal bar chart of feature importances.

    Args:
        crop: Crop name
        features: List of feature names
        importances: List of importance values
        output_path: Optional path to save the chart

    Returns:
        If output_path is None, returns base64 encoded image
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]

    # Colors based on sign
    colors = [COLORS['success'] if imp >= 0 else COLORS['danger'] for imp in importances]

    ax.barh(features, importances, color=colors, edgecolor='white', linewidth=0.5)

    # Add zero line
    ax.axvline(x=0, color='black', linewidth=0.5)

    ax.set_xlabel('Impact on Yield', fontsize=12)
    ax.set_title(f'{crop.replace("_", " ")} - Feature Importance',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_scenario_comparison_chart(historical_avg, predicted, crop, output_path=None):
    """
    Create a comparison chart between historical average and predicted yield.

    Args:
        historical_avg: Historical average yield
        predicted: Predicted yield
        crop: Crop name
        output_path: Optional path to save the chart

    Returns:
        If output_path is None, returns base64 encoded image
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ['Historical\nAverage', 'Current\nScenario']
    values = [historical_avg, predicted]

    colors = [COLORS['secondary'],
              COLORS['success'] if predicted >= historical_avg else COLORS['danger']]

    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Yield (t/ha)', fontsize=12)
    ax.set_title(f'{crop.replace("_", " ")} - Scenario Comparison',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_all_crops_comparison(predictions, output_path=None):
    """
    Create a comparison chart across all crops.

    Args:
        predictions: Dictionary of {crop: predicted_yield}
        output_path: Optional path to save the chart

    Returns:
        If output_path is None, returns base64 encoded image
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    crops = list(predictions.keys())
    predicted = [predictions[c] for c in crops]
    historical = [HISTORICAL_YIELDS.get(c, {}).get('yields', [7.0])[-1] for c in crops]

    x = np.arange(len(crops))
    width = 0.35

    bars1 = ax.bar(x - width/2, historical, width, label='Historical (2024)',
                  color=COLORS['secondary'], edgecolor='white')
    bars2 = ax.bar(x + width/2, predicted, width, label='Predicted',
                  color=COLORS['primary'], edgecolor='white')

    ax.set_ylabel('Yield (t/ha)', fontsize=12)
    ax.set_title('Yield Comparison Across All Crops', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ') for c in crops], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
