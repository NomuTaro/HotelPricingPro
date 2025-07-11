"""
価格最適化モジュール

このモジュールは、ホテルの価格最適化とシミュレーションを行う機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
import warnings

# 警告を非表示
warnings.filterwarnings('ignore')

# 定数定義
MAX_DAYS_BEFORE_STAY = 90


def ideal_booking_curve(t: np.ndarray, start_rate: float = 0.0, 
                       end_rate: float = 1.0, power: float = 1.5) -> np.ndarray:
    """
    理想のブッキングカーブを定義する関数。
    
    Args:
        t: 残り日数
        start_rate: 開始時の予約率（デフォルト: 0.0）
        end_rate: 最終日の予約率（デフォルト: 1.0）
        power: カーブの形状を制御する指数（デフォルト: 1.5）
    
    Returns:
        理想の予約率
    
    Formula:
        progress = (MAX_DAYS_BEFORE_STAY - t) / MAX_DAYS_BEFORE_STAY
        ideal_rate = start_rate + (end_rate - start_rate) * (progress ** power)
    """
    progress = (MAX_DAYS_BEFORE_STAY - t) / MAX_DAYS_BEFORE_STAY
    return start_rate + (end_rate - start_rate) * (progress ** power)


class PriceOptimizer:
    """
    価格最適化とシミュレーションを行うクラス。
    
    Attributes:
        analyzer: ブッキング分析クラスのインスタンス
        visualizer: 可視化クラスのインスタンス
    """

    def __init__(self, analyzer, visualizer):
        """
        初期化。
        
        Args:
            analyzer: ブッキング分析クラスのインスタンス
            visualizer: 可視化クラスのインスタンス
        """
        self.analyzer = analyzer
        self.visualizer = visualizer

    def find_optimal_price_for_day(self, target_sold: float, dow: int, is_weekend: int,
                                  days_left: int, booking_rate: float, stock: int,
                                  avg_price: float, price_range: Tuple[float, float]) -> float:
        """
        その日の目標販売数を達成するための最適価格を探索します。
        
        Args:
            target_sold: 目標販売数
            dow: 曜日 (0=月, 6=日)
            is_weekend: 週末フラグ (1=週末, 0=平日)
            days_left: 宿泊日までの残り日数
            booking_rate: 現在の予約率
            stock: 現在の在庫数
            avg_price: 平均価格（正規化用）
            price_range: 価格探索範囲 (最小価格, 最大価格)
        
        Returns:
            最適価格
        """
        best_price = np.mean(price_range)
        min_diff = float('inf')

        # 指定された価格範囲内で21点の候補を試す
        for price in np.linspace(price_range[0], price_range[1], 21):
            predicted_sold = self.analyzer.predict_daily_sold(
                price, dow, is_weekend, days_left, booking_rate, stock, avg_price
            )
            diff = abs(predicted_sold - target_sold)

            if diff < min_diff:
                min_diff = diff
                best_price = price

        return best_price

    def run_simulation(self, sim_target_group: pd.DataFrame, 
                      price_range: Tuple[float, float] = (10000, 30000)) -> Dict[str, any]:
        """
        指定されたグループに対して、価格最適化シミュレーションを実行します。
        
        Args:
            sim_target_group: シミュレーション対象のデータグループ
            price_range: 価格探索範囲 (最小価格, 最大価格)
        
        Returns:
            シミュレーション結果の辞書
        """
        print("\n--- Phase 3: 価格最適化シミュレーション開始 ---")

        # 初期パラメータの取得
        initial_stock = sim_target_group['initial_stock'].iloc[0]
        dow_stay = sim_target_group['dow_stay'].iloc[0]
        is_weekend_stay = sim_target_group['is_weekend_stay'].iloc[0]
        avg_price_hist = sim_target_group['price'].mean()

        # シミュレーションの初期化
        current_stock = initial_stock
        current_booking_rate = 0.0
        price_plan, booking_rate_sim = [], []
        revenue_sim = 0

        # 90日前からシミュレーション開始
        for t in range(MAX_DAYS_BEFORE_STAY, -1, -1):
            # 1. その日の目標予約率を理想カーブから取得
            ideal_rate_today = ideal_booking_curve(t)
            ideal_rate_tomorrow = ideal_booking_curve(t-1) if t > 0 else 1.0

            # 2. 目標販売数を計算
            target_sold_for_day = (ideal_rate_tomorrow - ideal_rate_today) * initial_stock

            # 3. 目標販売数を達成するための価格を探索
            price_for_day = self.find_optimal_price_for_day(
                target_sold=target_sold_for_day, 
                dow=dow_stay, 
                is_weekend=is_weekend_stay,
                days_left=t, 
                booking_rate=current_booking_rate, 
                stock=current_stock,
                avg_price=avg_price_hist, 
                price_range=price_range
            )

            # 4. 状態を更新
            sold_sim = self.analyzer.predict_daily_sold(
                price=price_for_day, 
                dow=dow_stay, 
                is_weekend=is_weekend_stay,
                days_left=t, 
                booking_rate=current_booking_rate, 
                stock=current_stock,
                avg_price=avg_price_hist
            )

            sold_sim = min(current_stock, sold_sim)  # 在庫以上に売れない
            current_stock -= sold_sim
            revenue_sim += sold_sim * price_for_day
            current_booking_rate = (initial_stock - current_stock) / initial_stock if initial_stock > 0 else 0

            price_plan.append({'days_before_stay': t, 'price': price_for_day})
            booking_rate_sim.append({'days_before_stay': t, 'booking_rate': current_booking_rate})

        price_plan_df = pd.DataFrame(price_plan)
        booking_rate_sim_df = pd.DataFrame(booking_rate_sim)

        print("✅ シミュレーション完了")
        revenue_hist = (sim_target_group['sold'] * sim_target_group['price']).sum()
        print(f"📊 結果比較:")
        print(f"  - 過去実績の売上: {revenue_hist:,.0f} 円")
        print(f"  - AI推奨価格プランによる予測最終売上: {revenue_sim:,.0f} 円")
        print(f"  - 改善率: {((revenue_sim - revenue_hist) / revenue_hist * 100):.1f}%")

        # 可視化
        self.visualizer.plot_simulation_results(
            historical_data=sim_target_group,
            simulation_booking_curve=booking_rate_sim_df,
            simulation_price_plan=price_plan_df,
            ideal_curve_func=ideal_booking_curve
        )

        return {
            'historical_revenue': revenue_hist,
            'simulated_revenue': revenue_sim,
            'improvement_rate': (revenue_sim - revenue_hist) / revenue_hist * 100,
            'price_plan': price_plan_df,
            'booking_curve': booking_rate_sim_df
        }

    def optimize_for_revenue_maximization(self, sim_target_group: pd.DataFrame,
                                        price_range: Tuple[float, float] = (10000, 30000),
                                        optimization_steps: int = 50) -> Dict[str, any]:
        """
        売上最大化を目的とした価格最適化を実行します。
        
        Args:
            sim_target_group: シミュレーション対象のデータグループ
            price_range: 価格探索範囲 (最小価格, 最大価格)
            optimization_steps: 最適化のステップ数
        
        Returns:
            最適化結果の辞書
        """
        print("\n--- 売上最大化価格最適化開始 ---")
        
        initial_stock = sim_target_group['initial_stock'].iloc[0]
        dow_stay = sim_target_group['dow_stay'].iloc[0]
        is_weekend_stay = sim_target_group['is_weekend_stay'].iloc[0]
        avg_price_hist = sim_target_group['price'].mean()
        
        best_revenue = 0
        best_price_plan = None
        best_booking_curve = None
        
        # 複数の価格戦略を試行
        for strategy in range(optimization_steps):
            # ランダムな価格戦略を生成
            base_price = np.random.uniform(price_range[0], price_range[1])
            price_variation = np.random.uniform(0.8, 1.2)
            
            current_stock = initial_stock
            current_booking_rate = 0.0
            revenue_total = 0
            price_plan = []
            booking_curve = []
            
            for t in range(MAX_DAYS_BEFORE_STAY, -1, -1):
                # 動的価格設定
                dynamic_price = base_price * price_variation * (1 + 0.1 * np.sin(t / 10))
                dynamic_price = max(price_range[0], min(price_range[1], dynamic_price))
                
                # 予約数予測
                sold = self.analyzer.predict_daily_sold(
                    price=dynamic_price,
                    dow=dow_stay,
                    is_weekend=is_weekend_stay,
                    days_left=t,
                    booking_rate=current_booking_rate,
                    stock=current_stock,
                    avg_price=avg_price_hist
                )
                
                sold = min(current_stock, sold)
                current_stock -= sold
                revenue_total += sold * dynamic_price
                current_booking_rate = (initial_stock - current_stock) / initial_stock if initial_stock > 0 else 0
                
                price_plan.append({'days_before_stay': t, 'price': dynamic_price})
                booking_curve.append({'days_before_stay': t, 'booking_rate': current_booking_rate})
            
            # 最良の結果を更新
            if revenue_total > best_revenue:
                best_revenue = revenue_total
                best_price_plan = pd.DataFrame(price_plan)
                best_booking_curve = pd.DataFrame(booking_curve)
        
        print(f"✅ 最適化完了")
        print(f"📊 最適化結果:")
        print(f"  - 最適売上: {best_revenue:,.0f} 円")
        
        return {
            'optimal_revenue': best_revenue,
            'price_plan': best_price_plan,
            'booking_curve': best_booking_curve
        }

    def sensitivity_analysis(self, sim_target_group: pd.DataFrame,
                           base_price_range: Tuple[float, float] = (10000, 30000),
                           sensitivity_factor: float = 0.1) -> Dict[str, any]:
        """
        価格感度分析を実行します。
        
        Args:
            sim_target_group: シミュレーション対象のデータグループ
            base_price_range: 基本価格範囲
            sensitivity_factor: 感度分析の変動係数
        
        Returns:
            感度分析結果の辞書
        """
        print("\n--- 価格感度分析開始 ---")
        
        base_price = np.mean(base_price_range)
        price_variations = np.linspace(1 - sensitivity_factor, 1 + sensitivity_factor, 21)
        revenue_results = []
        
        for variation in price_variations:
            test_price = base_price * variation
            
            # 簡易シミュレーション
            total_revenue = 0
            current_stock = sim_target_group['initial_stock'].iloc[0]
            
            for t in range(MAX_DAYS_BEFORE_STAY, -1, -1):
                if current_stock <= 0:
                    break
                    
                sold = self.analyzer.predict_daily_sold(
                    price=test_price,
                    dow=sim_target_group['dow_stay'].iloc[0],
                    is_weekend=sim_target_group['is_weekend_stay'].iloc[0],
                    days_left=t,
                    booking_rate=0.5,  # 仮定
                    stock=current_stock,
                    avg_price=base_price
                )
                
                sold = min(current_stock, sold)
                total_revenue += sold * test_price
                current_stock -= sold
            
            revenue_results.append({
                'price_variation': variation,
                'price': test_price,
                'revenue': total_revenue
            })
        
        sensitivity_df = pd.DataFrame(revenue_results)
        
        print(f"✅ 感度分析完了")
        print(f"📊 最適価格変動率: {sensitivity_df.loc[sensitivity_df['revenue'].idxmax(), 'price_variation']:.3f}")
        
        return {
            'sensitivity_data': sensitivity_df,
            'optimal_variation': sensitivity_df.loc[sensitivity_df['revenue'].idxmax(), 'price_variation'],
            'optimal_price': sensitivity_df.loc[sensitivity_df['revenue'].idxmax(), 'price']
        } 