"""
データ前処理モジュール

このモジュールは、ホテル予約データの前処理と準備を行う機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings

# 警告を非表示
warnings.filterwarnings('ignore')

# 定数定義
MAX_DAYS_BEFORE_STAY = 90


class DataPreprocessor:
    """
    データの前処理と準備を行うクラス。
    
    Attributes:
        file_path (str): データファイルのパス
        df (pd.DataFrame): 処理済みのデータフレーム
    """

    def __init__(self, file_path: str):
        """
        初期化。
        
        Args:
            file_path: データファイルのパス
        """
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> bool:
        """
        データファイルを読み込みます。
        
        Returns:
            読み込み成功時True
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print("✅ データ読み込み完了")
            print(f"📊 データ件数: {len(self.df):,}件")
            print(f"📋 カラム数: {len(self.df.columns)}個")
            return True
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return False

    def preprocess(self) -> Optional[pd.DataFrame]:
        """
        データの前処理（型変換、特徴量生成など）を実行します。
        
        生成される特徴量:
        - sold: １日あたりの販売数
        - revenue: １日あたりの売上
        - cumulative_sold: 累計販売数
        - initial_stock: 予約日90日前時点での初期在庫数
        - booking_rate: 予約率
        - days_before_stay: 宿泊日までの残り日数
        - dow_stay: 宿泊日の曜日 (0=月, 6=日)
        - is_weekend_stay: 宿泊日が週末 (土日) なら1、それ以外は0
        
        Returns:
            処理済みのデータフレーム
        """
        if self.df is None:
            print("❌ データが読み込まれていません。")
            return None

        print("\n🔄 データ前処理開始")
        
        # 型変換
        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
        self.df['date'] = pd.to_datetime(self.df['date'])

        # sold（成約数）とrevenue（売上）の計算
        self.df = self.df.sort_values(['hotel_id', 'plan_id', 'room_type_id', 'date', 'created_at'])
        self.df['stock_shift'] = self.df.groupby(['hotel_id', 'plan_id', 'room_type_id', 'date'])['stock'].shift(1)
        self.df['sold'] = (self.df['stock_shift'] - self.df['stock']).clip(lower=0).fillna(0)
        self.df['revenue'] = self.df['sold'] * self.df['price']

        # 特徴量生成
        group_cols = ['hotel_id', 'plan_id', 'room_type_id', 'date']
        self.df['cumulative_sold'] = self.df.groupby(group_cols)['sold'].cumsum()

        # initial_stockの計算
        group_stats = self.df.groupby(group_cols).agg(
            last_stock=('stock', 'last'),
            total_sold=('sold', 'sum')
        ).reset_index()
        group_stats['initial_stock'] = group_stats['last_stock'] + group_stats['total_sold']
        self.df = pd.merge(self.df, group_stats[group_cols + ['initial_stock']], on=group_cols, how='left')

        # 予約率の計算
        self.df['booking_rate'] = np.where(
            self.df['initial_stock'] > 0, 
            self.df['cumulative_sold'] / self.df['initial_stock'], 
            0
        ).clip(0, 1)
        
        # 残り日数の計算
        self.df['days_before_stay'] = (self.df['date'] - self.df['created_at']).dt.days

        # 曜日などの特徴量
        self.df['dow_stay'] = self.df['date'].dt.dayofweek
        self.df['is_weekend_stay'] = (self.df['dow_stay'] >= 5).astype(int)
        
        # 月次特徴量
        self.df['month_stay'] = self.df['date'].dt.month
        self.df['is_holiday'] = self.df['dow_stay'].isin([5, 6]).astype(int)

        # 不要なカラムを削除
        self.df.drop(columns=['stock_shift'], inplace=True, errors='ignore')
        
        print("✅ データ前処理完了")
        print(f"📊 処理後データ件数: {len(self.df):,}件")
        
        return self.df

    def filter_for_analysis(self, min_initial_stock: int = 10, min_effective_days: int = 5) -> pd.DataFrame:
        """
        分析対象のデータをフィルタリングします。
        
        Args:
            min_initial_stock: 最小初期在庫数（デフォルト: 10）
            min_effective_days: 最小有効日数（デフォルト: 5）
        
        Returns:
            フィルタリング済みのデータフレーム
        """
        print(f"\n🔍 分析対象のフィルタリング開始")

        # 初期在庫でフィルタリング
        df_filtered = self.df[self.df['initial_stock'] >= min_initial_stock].copy()
        print(f"📦 初期在庫{min_initial_stock}以上にフィルタリング後: "
              f"{len(df_filtered.drop_duplicates(subset=['hotel_id', 'plan_id', 'room_type_id', 'date']))} グループ")

        # 有効日数でフィルタリング
        group_cols = ['hotel_id', 'plan_id', 'room_type_id', 'date']
        effective_days_count = df_filtered.groupby(group_cols)['created_at'].nunique().reset_index()
        effective_days_count.rename(columns={'created_at': 'effective_days'}, inplace=True)

        valid_groups = effective_days_count[effective_days_count['effective_days'] >= min_effective_days]
        print(f"📅 有効日数{min_effective_days}日以上にフィルタリング後: {len(valid_groups)} グループが分析対象")

        df_for_fitting = df_filtered.merge(valid_groups[group_cols], on=group_cols)
        print(f"📊 最終的な分析対象データ件数: {len(df_for_fitting):,}件")

        # 最大日数でフィルタリング
        df_for_fitting = df_for_fitting[df_for_fitting['days_before_stay'] <= MAX_DAYS_BEFORE_STAY]
        print(f"⏰ 宿泊日{MAX_DAYS_BEFORE_STAY}日前のデータに絞り込み後: {len(df_for_fitting):,}件")

        return df_for_fitting

    def get_data_summary(self) -> None:
        """データの概要統計を表示します。"""
        if self.df is None:
            print("❌ データが読み込まれていません。")
            return
        
        print("\n📊 データ概要統計")
        print("=" * 50)
        
        # 基本統計
        print(f"総データ件数: {len(self.df):,}件")
        print(f"ホテル数: {self.df['hotel_id'].nunique()}軒")
        print(f"プラン数: {self.df['plan_id'].nunique()}種類")
        print(f"部屋タイプ数: {self.df['room_type_id'].nunique()}種類")
        
        # 価格統計
        print(f"\n💰 価格統計:")
        print(f"平均価格: {self.df['price'].mean():,.0f}円")
        print(f"最小価格: {self.df['price'].min():,.0f}円")
        print(f"最大価格: {self.df['price'].max():,.0f}円")
        
        # 在庫統計
        print(f"\n📦 在庫統計:")
        print(f"平均在庫: {self.df['stock'].mean():.1f}室")
        print(f"平均初期在庫: {self.df['initial_stock'].mean():.1f}室")
        
        # 予約統計
        print(f"\n📈 予約統計:")
        print(f"総販売数: {self.df['sold'].sum():,}室")
        print(f"総売上: {self.df['revenue'].sum():,.0f}円")
        print(f"平均予約率: {self.df['booking_rate'].mean():.2%}") 