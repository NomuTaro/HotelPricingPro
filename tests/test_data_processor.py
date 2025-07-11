"""
データ処理モジュールのテスト

このモジュールは、DataPreprocessorクラスのテストを提供します。
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessorクラスのテストケース"""

    def setUp(self):
        """テスト前の準備"""
        # テスト用データの作成
        self.test_data = self._create_test_data()
        
        # 一時ファイルの作成
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # DataPreprocessorインスタンスの作成
        self.processor = DataPreprocessor(self.temp_file.name)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ファイルの削除
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def _create_test_data(self):
        """テスト用データを作成"""
        np.random.seed(42)
        
        data = []
        base_date = datetime.now()
        
        for hotel_id in range(1, 3):
            for plan_id in range(1, 3):
                for room_type_id in range(1, 3):
                    for date_idx in range(5):
                        stay_date = base_date + timedelta(days=date_idx + 30)
                        initial_stock = 20
                        current_stock = initial_stock
                        
                        for days_before in range(90, -1, -10):
                            booking_date = stay_date - timedelta(days=days_before)
                            
                            # 価格設定
                            base_price = 15000 + hotel_id * 1000 + plan_id * 500
                            price = int(base_price * (1 + 0.1 * np.sin(days_before / 10)))
                            
                            # 在庫更新
                            if days_before < 90:
                                sold = np.random.randint(0, 3)
                                current_stock = max(0, current_stock - sold)
                            
                            data.append({
                                'hotel_id': hotel_id,
                                'plan_id': plan_id,
                                'room_type_id': room_type_id,
                                'date': stay_date.strftime('%Y-%m-%d'),
                                'created_at': booking_date.strftime('%Y-%m-%d %H:%M:%S'),
                                'price': price,
                                'stock': current_stock
                            })
        
        return pd.DataFrame(data)

    def test_load_data(self):
        """データ読み込みのテスト"""
        result = self.processor.load_data()
        self.assertTrue(result)
        self.assertIsNotNone(self.processor.df)
        self.assertEqual(len(self.processor.df), len(self.test_data))

    def test_preprocess(self):
        """データ前処理のテスト"""
        # データ読み込み
        self.processor.load_data()
        
        # 前処理実行
        result = self.processor.preprocess()
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertIn('sold', result.columns)
        self.assertIn('revenue', result.columns)
        self.assertIn('cumulative_sold', result.columns)
        self.assertIn('initial_stock', result.columns)
        self.assertIn('booking_rate', result.columns)
        self.assertIn('days_before_stay', result.columns)
        self.assertIn('dow_stay', result.columns)
        self.assertIn('is_weekend_stay', result.columns)

    def test_filter_for_analysis(self):
        """分析対象フィルタリングのテスト"""
        # データ読み込みと前処理
        self.processor.load_data()
        self.processor.preprocess()
        
        # フィルタリング実行
        result = self.processor.filter_for_analysis(min_initial_stock=10, min_effective_days=3)
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        
        # フィルタリング条件の確認
        self.assertTrue(all(result['initial_stock'] >= 10))
        self.assertTrue(all(result['days_before_stay'] <= 90))

    def test_get_data_summary(self):
        """データ概要統計のテスト"""
        # データ読み込みと前処理
        self.processor.load_data()
        self.processor.preprocess()
        
        # 概要統計の実行（エラーが発生しないことを確認）
        try:
            self.processor.get_data_summary()
        except Exception as e:
            self.fail(f"get_data_summary() raised {e} unexpectedly!")

    def test_invalid_file_path(self):
        """無効なファイルパスのテスト"""
        invalid_processor = DataPreprocessor("nonexistent_file.csv")
        result = invalid_processor.load_data()
        self.assertFalse(result)

    def test_empty_data(self):
        """空データのテスト"""
        # 空のデータフレームでテスト
        empty_df = pd.DataFrame()
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            processor = DataPreprocessor(temp_file.name)
            processor.load_data()
            result = processor.preprocess()
            self.assertIsNone(result)
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


if __name__ == '__main__':
    unittest.main() 