"""
Hotel Pricing Pro - ホテル価格最適化システム

This package provides tools for hotel pricing optimization using machine learning.
"""

__version__ = "1.0.0"
__author__ = "Hotel Pricing Pro Team"
__email__ = "team@hotelpricingpro.com"

# モジュールが存在する場合のみインポート
try:
    from .data_processor import DataPreprocessor
    from .booking_analyzer import BookingCurveAnalyzer
    from .price_optimizer import PriceOptimizer
    from .visualizer import Visualizer

    __all__ = [
        "DataPreprocessor",
        "BookingCurveAnalyzer", 
        "PriceOptimizer",
        "Visualizer"
    ]
except ImportError:
    # モジュールがまだ作成されていない場合
    __all__ = [] 