"""
ブッキング分析モジュール

このモジュールは、ブッキングカーブの分析と予測モデルの構築を行う機能を提供します。
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, Dict, Any, Tuple
import warnings

# 警告を非表示
warnings.filterwarnings('ignore')

# 定数定義
MAX_DAYS_BEFORE_STAY = 90
RANDOM_SEED = 42


def booking_curve_func(t: np.ndarray, r0: float, alpha: float, t0: float) -> np.ndarray:
    """
    ブッキングカーブを表現する指数関数。
    
    Args:
        t: 残り日数
        r0: 初期予約率
        alpha: 予約強度パラメータ（指数の係数）
        t0: 時間オフセット
    
    Returns:
        予約率の予測値
    
    Formula:
        r = r0 * exp(alpha * (MAX_DAYS_BEFORE_STAY - t - t0))
    """
    return r0 * np.exp(alpha * (MAX_DAYS_BEFORE_STAY - t - t0))


class BookingCurveAnalyzer:
    """
    ブッキングカーブの分析と予測モデルの構築・管理を行うクラス。
    
    Attributes:
        alpha_df (pd.DataFrame): alphaパラメータの分析結果
        model (lgb.LGBMRegressor): 学習済みの予測モデル
        features (list): モデルが使用する特徴量のリスト
    """

    def __init__(self, df_for_fitting: Optional[pd.DataFrame] = None):
        """
        初期化。
        
        Args:
            df_for_fitting: 分析用のデータフレーム
        """
        self.df = df_for_fitting
        self.alpha_df: Optional[pd.DataFrame] = None
        self.model: Optional[lgb.LGBMRegressor] = None
        self.features = [
            'price_normalized', 'dow_stay', 'is_weekend_stay', 
            'days_before_stay', 'current_booking_rate', 'stock'
        ]

    def calculate_actual_alphas(self, df_for_fitting: pd.DataFrame) -> pd.DataFrame:
        """
        実績データからカーブフィッティングを行い、各グループのalphaを算出します。
        
        Args:
            df_for_fitting: 分析用のデータフレーム
        
        Returns:
            alphaパラメータの分析結果
        """
        print("\n--- Phase 1: 実績alphaの算出開始 ---")
        alpha_results = []
        group_cols = ['hotel_id', 'plan_id', 'room_type_id', 'date']
        grouped = df_for_fitting.groupby(group_cols)

        for name, group in grouped:
            if group['sold'].sum() == 0 or len(group) < 3:
                continue

            x_data = group['days_before_stay'].values
            y_data = group['booking_rate'].values

            try:
                # 最終予約率が低い場合は、フィッティングが不安定になりやすいため、境界を調整
                final_booking_rate = y_data[-1]
                # パラメータの初期値と境界を設定 (r0, alpha, t0)
                initial_params = [0.01, 0.05, 0]
                bounds = (
                    [0, 0, -np.inf], 
                    [max(0.01, final_booking_rate), 5, np.inf]
                )

                params, _ = curve_fit(
                    booking_curve_func, x_data, y_data, 
                    p0=initial_params, bounds=bounds, maxfev=5000
                )
                r0, alpha, t0 = params

                alpha_results.append({
                    'hotel_id': name[0], 'plan_id': name[1], 
                    'room_type_id': name[2], 'date': name[3],
                    'alpha': alpha, 'r0': r0, 't0': t0,
                    'initial_price': group['price'].iloc[0],
                    'dow_stay': group['dow_stay'].iloc[0],
                    'is_weekend_stay': group['is_weekend_stay'].iloc[0],
                    'initial_stock': group['initial_stock'].iloc[0]
                })
            except RuntimeError:
                continue

        self.alpha_df = pd.DataFrame(alpha_results)
        print(f"✅ alphaの算出完了。対象となった予約グループ数: {len(self.alpha_df)}件")
        return self.alpha_df

    def build_daily_sold_predictor(self) -> bool:
        """
        １日あたりの予約販売数(sold)を予測する機械学習モデルを構築します。
        
        Returns:
            モデル構築成功時True
        """
        if self.df is None:
            print("❌ データが読み込まれていません。")
            return False

        print("\n--- Phase 2: 日次予約数予測モデルの学習開始 ---")

        # 特徴量エンジニアリング
        # 価格を正規化（部屋タイプごとの平均価格で割る）
        avg_price_per_group = self.df.groupby(['hotel_id', 'plan_id', 'room_type_id'])['price'].mean().to_dict()
        key_cols = ['hotel_id', 'plan_id', 'room_type_id']
        self.df['avg_price_group'] = self.df[key_cols].apply(tuple, axis=1).map(avg_price_per_group)
        self.df['price_normalized'] = self.df['price'] / self.df['avg_price_group']

        # その日の開始時点での予約率（前日までの予約率）
        self.df['current_booking_rate'] = self.df.groupby(['hotel_id', 'plan_id', 'room_type_id', 'date'])['booking_rate'].shift(1).fillna(0)

        # 目的変数は「その日に何件売れたか(sold)」
        target = 'sold'

        X = self.df[self.features]
        y = self.df[target]

        # 欠損値を処理
        X = X.fillna(0)
        y = y.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # LightGBMパラメータ
        lgb_params = {
            'objective': 'poisson',  # カウントデータなのでポアソン回帰が適している
            'metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': RANDOM_SEED
        }
        
        self.model = lgb.LGBMRegressor(**lgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        print("✅ 日次予約数予測モデルの学習完了")
        
        # モデル評価
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 モデル評価:")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        return True

    def predict_daily_sold(self, price: float, dow: int, is_weekend: int, 
                          days_left: int, booking_rate: float, stock: int, 
                          avg_price: float) -> float:
        """
        学習済みモデルを使い、特定の条件下での日次予約数を予測します。
        
        Args:
            price: 価格
            dow: 曜日 (0=月, 6=日)
            is_weekend: 週末フラグ (1=週末, 0=平日)
            days_left: 宿泊日までの残り日数
            booking_rate: 現在の予約率
            stock: 現在の在庫数
            avg_price: 平均価格（正規化用）
        
        Returns:
            予測される日次予約数
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")

        price_norm = price / avg_price if avg_price > 0 else 1

        feature_values = [price_norm, dow, is_weekend, days_left, booking_rate, stock]
        df_pred = pd.DataFrame([feature_values], columns=self.features)

        predicted_sold = self.model.predict(df_pred)[0]
        return max(0, predicted_sold)  # 予測値がマイナスにならないようにクリップ

    def get_feature_importance(self) -> pd.DataFrame:
        """
        特徴量の重要度を取得します。
        
        Returns:
            特徴量重要度のデータフレーム
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def cross_validate_model(self, cv_folds: int = 5) -> Dict[str, float]:
        """
        クロスバリデーションでモデルの性能を評価します。
        
        Args:
            cv_folds: クロスバリデーションの分割数
        
        Returns:
            評価指標の辞書
        """
        if self.df is None:
            raise ValueError("データが読み込まれていません。")
        
        # 特徴量エンジニアリング（build_daily_sold_predictorと同じ処理）
        avg_price_per_group = self.df.groupby(['hotel_id', 'plan_id', 'room_type_id'])['price'].mean().to_dict()
        key_cols = ['hotel_id', 'plan_id', 'room_type_id']
        self.df['avg_price_group'] = self.df[key_cols].apply(tuple, axis=1).map(avg_price_per_group)
        self.df['price_normalized'] = self.df['price'] / self.df['avg_price_group']
        self.df['current_booking_rate'] = self.df.groupby(['hotel_id', 'plan_id', 'room_type_id', 'date'])['booking_rate'].shift(1).fillna(0)

        X = self.df[self.features].fillna(0)
        y = self.df['sold'].fillna(0)

        # クロスバリデーション
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        
        lgb_params = {
            'objective': 'poisson',
            'metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'seed': RANDOM_SEED
        }
        
        model = lgb.LGBMRegressor(**lgb_params)
        
        # 各指標でクロスバリデーション
        rmse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        results = {
            'rmse_mean': -rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
        
        print(f"📊 クロスバリデーション結果 ({cv_folds}分割):")
        print(f"  - RMSE: {results['rmse_mean']:.4f} (±{results['rmse_std']:.4f})")
        print(f"  - MAE: {results['mae_mean']:.4f} (±{results['mae_std']:.4f})")
        print(f"  - R²: {results['r2_mean']:.4f} (±{results['r2_std']:.4f})")
        
        return results 