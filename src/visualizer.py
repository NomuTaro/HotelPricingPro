"""
可視化モジュール

このモジュールは、分析結果の可視化を行う機能を提供します。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Callable, Dict, Any
import warnings

# 警告を非表示
warnings.filterwarnings('ignore')

# 定数定義
MAX_DAYS_BEFORE_STAY = 90


class Visualizer:
    """
    結果を可視化するクラス。
    """

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初期化。
        
        Args:
            style: matplotlibのスタイル設定
        """
        plt.style.use(style)
        self.colors = {
            'historical': '#2E86AB',
            'simulation': '#A23B72',
            'ideal': '#F18F01',
            'price': '#C73E1D',
            'background': '#F8F9FA'
        }

    def plot_simulation_results(self, historical_data: pd.DataFrame,
                               simulation_booking_curve: pd.DataFrame,
                               simulation_price_plan: pd.DataFrame,
                               ideal_curve_func: Callable) -> None:
        """
        シミュレーション結果をグラフに描画します。
        
        Args:
            historical_data: 実績データ
            simulation_booking_curve: シミュレーション予約率データ
            simulation_price_plan: シミュレーション価格データ
            ideal_curve_func: 理想カーブ関数
        """
        print("\n--- Phase 4: 評価と可視化 ---")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 1. 予約率の比較グラフ
        ax1.set_xlabel('宿泊日までの残り日数')
        ax1.set_ylabel('予約率 (%)', color='black')
        ax1.set_ylim(0, 105)

        # 実績カーブ
        ax1.plot(historical_data['days_before_stay'], 
                historical_data['booking_rate'] * 100,
                label='実績カーブ', color=self.colors['historical'], 
                linestyle='--', marker='o', markersize=3, zorder=2)

        # AIシミュレーションカーブ
        ax1.plot(simulation_booking_curve['days_before_stay'], 
                simulation_booking_curve['booking_rate'] * 100,
                label='AI推奨プランカーブ', color=self.colors['simulation'], 
                linewidth=2.5, zorder=3)

        # 理想カーブ
        t_range = np.arange(MAX_DAYS_BEFORE_STAY, -1, -1)
        ideal_curve = [ideal_curve_func(t) * 100 for t in t_range]
        ax1.plot(t_range, ideal_curve, label='理想カーブ', 
                color=self.colors['ideal'], linestyle=':', linewidth=2, zorder=1)

        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
        ax1.legend(loc='upper left')
        ax1.set_title('ブッキングカーブ比較', fontsize=14, fontweight='bold')
        ax1.invert_xaxis()

        # 2. 価格比較グラフ
        ax2.set_xlabel('宿泊日までの残り日数')
        ax2.set_ylabel('価格 (円)', color='green')

        # 実績価格
        ax2.plot(historical_data['days_before_stay'], historical_data['price'],
                label='実績価格', color=self.colors['historical'], 
                linestyle='--', marker='x', markersize=4, zorder=2)

        # AI推奨価格
        ax2.plot(simulation_price_plan['days_before_stay'], 
                simulation_price_plan['price'],
                label='AI推奨価格', color=self.colors['simulation'], 
                linewidth=2, zorder=3)

        ax2.tick_params(axis='y', labelcolor='green')
        ax2.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
        ax2.legend(loc='upper left')
        ax2.set_title('価格推移比較', fontsize=14, fontweight='bold')
        ax2.invert_xaxis()

        # タイトル情報
        title_info = historical_data.iloc[0]
        fig.suptitle(
            f"ホテル価格最適化シミュレーション結果\n"
            f"ホテルID: {title_info['hotel_id']} | "
            f"プランID: {title_info['plan_id']} | "
            f"部屋ID: {title_info['room_type_id']} | "
            f"宿泊日: {title_info['date'].date()}",
            fontsize=16, fontweight='bold'
        )

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """
        特徴量重要度を可視化します。
        
        Args:
            importance_df: 特徴量重要度のデータフレーム
        """
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                       color=colors, alpha=0.8)
        
        # バーの上に値を表示
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.xlabel('重要度')
        plt.title('特徴量重要度', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_booking_curve_analysis(self, alpha_df: pd.DataFrame) -> None:
        """
        ブッキングカーブ分析結果を可視化します。
        
        Args:
            alpha_df: alphaパラメータの分析結果
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Alpha分布
        ax1.hist(alpha_df['alpha'], bins=30, alpha=0.7, color=self.colors['historical'])
        ax1.set_xlabel('Alpha値')
        ax1.set_ylabel('頻度')
        ax1.set_title('Alpha分布')
        ax1.grid(True, alpha=0.3)

        # 2. 価格とAlphaの関係
        ax2.scatter(alpha_df['initial_price'], alpha_df['alpha'], 
                   alpha=0.6, color=self.colors['simulation'])
        ax2.set_xlabel('初期価格')
        ax2.set_ylabel('Alpha値')
        ax2.set_title('価格とAlphaの関係')
        ax2.grid(True, alpha=0.3)

        # 3. 曜日別Alpha分布
        dow_labels = ['月', '火', '水', '木', '金', '土', '日']
        dow_alpha = [alpha_df[alpha_df['dow_stay'] == i]['alpha'].mean() 
                    for i in range(7)]
        ax3.bar(dow_labels, dow_alpha, color=self.colors['ideal'], alpha=0.7)
        ax3.set_xlabel('曜日')
        ax3.set_ylabel('平均Alpha値')
        ax3.set_title('曜日別Alpha分布')
        ax3.grid(True, alpha=0.3)

        # 4. 在庫数とAlphaの関係
        ax4.scatter(alpha_df['initial_stock'], alpha_df['alpha'], 
                   alpha=0.6, color=self.colors['price'])
        ax4.set_xlabel('初期在庫数')
        ax4.set_ylabel('Alpha値')
        ax4.set_title('在庫数とAlphaの関係')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_revenue_comparison(self, historical_revenue: float, 
                               simulated_revenue: float) -> None:
        """
        売上比較を可視化します。
        
        Args:
            historical_revenue: 実績売上
            simulated_revenue: シミュレーション売上
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['実績売上', 'AI推奨売上']
        revenues = [historical_revenue, simulated_revenue]
        colors = [self.colors['historical'], self.colors['simulation']]
        
        bars = ax.bar(categories, revenues, color=colors, alpha=0.8)
        
        # バーの上に値を表示
        for bar, revenue in zip(bars, revenues):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{revenue:,.0f}円', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12)
        
        # 改善率を表示
        improvement_rate = ((simulated_revenue - historical_revenue) / historical_revenue) * 100
        ax.text(0.5, 0.95, f'改善率: {improvement_rate:+.1f}%', 
               transform=ax.transAxes, ha='center', va='top',
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_ylabel('売上 (円)')
        ax.set_title('売上比較', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_price_sensitivity(self, sensitivity_data: pd.DataFrame) -> None:
        """
        価格感度分析結果を可視化します。
        
        Args:
            sensitivity_data: 感度分析データ
        """
        plt.figure(figsize=(12, 8))
        
        # 価格変動率と売上の関係
        plt.plot(sensitivity_data['price_variation'], sensitivity_data['revenue'], 
                marker='o', linewidth=2, markersize=6, color=self.colors['simulation'])
        
        # 最適点を強調
        optimal_idx = sensitivity_data['revenue'].idxmax()
        optimal_variation = sensitivity_data.loc[optimal_idx, 'price_variation']
        optimal_revenue = sensitivity_data.loc[optimal_idx, 'revenue']
        
        plt.scatter(optimal_variation, optimal_revenue, 
                   color='red', s=100, zorder=5, label=f'最適点\n変動率: {optimal_variation:.3f}')
        
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='基準価格')
        
        plt.xlabel('価格変動率 (1.0 = 基準価格)')
        plt.ylabel('予測売上 (円)')
        plt.title('価格感度分析', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_dashboard(self, historical_data: pd.DataFrame,
                        simulation_results: Dict[str, Any]) -> None:
        """
        総合的なダッシュボードを作成します。
        
        Args:
            historical_data: 実績データ
            simulation_results: シミュレーション結果
        """
        fig = plt.figure(figsize=(20, 12))
        
        # サブプロットの配置
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 売上比較
        ax1 = fig.add_subplot(gs[0, :2])
        categories = ['実績売上', 'AI推奨売上']
        revenues = [simulation_results['historical_revenue'], 
                   simulation_results['simulated_revenue']]
        colors = [self.colors['historical'], self.colors['simulation']]
        ax1.bar(categories, revenues, color=colors, alpha=0.8)
        ax1.set_ylabel('売上 (円)')
        ax1.set_title('売上比較')
        ax1.grid(True, alpha=0.3)
        
        # 2. 改善率
        ax2 = fig.add_subplot(gs[0, 2:])
        improvement_rate = simulation_results['improvement_rate']
        ax2.pie([100, improvement_rate], labels=['基準', f'改善\n{improvement_rate:+.1f}%'],
                colors=['lightgray', 'lightgreen'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('改善率')
        
        # 3. ブッキングカーブ比較
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(historical_data['days_before_stay'], 
                historical_data['booking_rate'] * 100,
                label='実績', color=self.colors['historical'], marker='o')
        ax3.plot(simulation_results['booking_curve']['days_before_stay'],
                simulation_results['booking_curve']['booking_rate'] * 100,
                label='AI推奨', color=self.colors['simulation'], linewidth=2)
        ax3.set_xlabel('宿泊日までの残り日数')
        ax3.set_ylabel('予約率 (%)')
        ax3.set_title('ブッキングカーブ比較')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
        
        # 4. 価格推移
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(historical_data['days_before_stay'], historical_data['price'],
                label='実績価格', color=self.colors['historical'], marker='x')
        ax4.plot(simulation_results['price_plan']['days_before_stay'],
                simulation_results['price_plan']['price'],
                label='AI推奨価格', color=self.colors['simulation'], linewidth=2)
        ax4.set_xlabel('宿泊日までの残り日数')
        ax4.set_ylabel('価格 (円)')
        ax4.set_title('価格推移比較')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()
        
        # タイトル
        title_info = historical_data.iloc[0]
        fig.suptitle(
            f"ホテル価格最適化ダッシュボード\n"
            f"ホテルID: {title_info['hotel_id']} | "
            f"プランID: {title_info['plan_id']} | "
            f"宿泊日: {title_info['date'].date()}",
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show() 