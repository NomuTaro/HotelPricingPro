"""
ホテル価格最適化システム - メイン実行ファイル

このモジュールは、ホテル価格最適化システムのメイン実行機能を提供します。
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor import DataPreprocessor
from src.booking_analyzer import BookingCurveAnalyzer
from src.price_optimizer import PriceOptimizer
from src.visualizer import Visualizer


def main(data_path: str, output_dir: Optional[str] = None, 
         min_stock: int = 10, min_days: int = 5) -> None:
    """
    メイン処理を実行します。
    
    Args:
        data_path: データファイルのパス
        output_dir: 出力ディレクトリ（オプション）
        min_stock: 最小初期在庫数
        min_days: 最小有効日数
    """
    print("🏨 ホテル価格最適化システム")
    print("=" * 50)
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = project_root / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # === データ準備 ===
        print("\n📊 データ準備フェーズ")
        data_processor = DataPreprocessor(data_path)
        
        if not data_processor.load_data():
            print("❌ データ読み込みに失敗しました。")
            return
        
        df_processed = data_processor.preprocess()
        if df_processed is None:
            print("❌ データ前処理に失敗しました。")
            return
        
        # データ概要の表示
        data_processor.get_data_summary()
        
        # 分析対象データのフィルタリング
        df_for_analysis = data_processor.filter_for_analysis(
            min_initial_stock=min_stock, 
            min_effective_days=min_days
        )
        
        if df_for_analysis.empty:
            print("❌ 分析対象となるデータがありませんでした。フィルタリング条件を緩和してください。")
            return
        
        # === モデル学習 ===
        print("\n🤖 モデル学習フェーズ")
        analyzer = BookingCurveAnalyzer(df_for_analysis)
        
        if not analyzer.build_daily_sold_predictor():
            print("❌ モデルの学習に失敗しました。")
            return
        
        # クロスバリデーション
        print("\n📈 モデル評価")
        cv_results = analyzer.cross_validate_model()
        
        # 特徴量重要度の表示
        importance_df = analyzer.get_feature_importance()
        print(f"\n🔍 特徴量重要度:")
        for _, row in importance_df.iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")
        
        # === シミュレーションと最適化 ===
        print("\n🎯 シミュレーションフェーズ")
        visualizer = Visualizer()
        optimizer = PriceOptimizer(analyzer, visualizer)
        
        # シミュレーション対象のグループを選択
        group_cols = ['hotel_id', 'plan_id', 'room_type_id', 'date']
        first_group_key = df_for_analysis[group_cols].iloc[0].to_dict()
        
        sim_target_group = df_for_analysis[
            (df_for_analysis['hotel_id'] == first_group_key['hotel_id']) &
            (df_for_analysis['plan_id'] == first_group_key['plan_id']) &
            (df_for_analysis['room_type_id'] == first_group_key['room_type_id']) &
            (df_for_analysis['date'] == first_group_key['date'])
        ].copy().sort_values('days_before_stay', ascending=False)
        
        # 価格の探索範囲を設定
        avg_price = sim_target_group['price'].mean()
        price_range = (avg_price * 0.7, avg_price * 1.5)
        
        # シミュレーション実行
        simulation_results = optimizer.run_simulation(sim_target_group, price_range=price_range)
        
        # 価格感度分析
        print("\n📊 価格感度分析")
        sensitivity_results = optimizer.sensitivity_analysis(sim_target_group, price_range)
        
        # 総合ダッシュボード
        print("\n📋 総合ダッシュボード")
        visualizer.create_dashboard(sim_target_group, simulation_results)
        
        # 結果の保存
        print(f"\n💾 結果を保存中: {output_dir}")
        
        # 特徴量重要度の保存
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        
        # シミュレーション結果の保存
        simulation_results['price_plan'].to_csv(output_dir / "optimal_price_plan.csv", index=False)
        simulation_results['booking_curve'].to_csv(output_dir / "simulated_booking_curve.csv", index=False)
        
        # 感度分析結果の保存
        sensitivity_results['sensitivity_data'].to_csv(output_dir / "price_sensitivity.csv", index=False)
        
        print("✅ 処理完了！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data(output_path: str) -> None:
    """
    サンプルデータを作成します。
    
    Args:
        output_path: 出力ファイルパス
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("📝 サンプルデータを作成中...")
    
    # サンプルデータの生成
    np.random.seed(42)
    
    # 基本パラメータ
    n_hotels = 3
    n_plans = 2
    n_room_types = 2
    n_dates = 30
    n_days_before = 90
    
    data = []
    
    for hotel_id in range(1, n_hotels + 1):
        for plan_id in range(1, n_plans + 1):
            for room_type_id in range(1, n_room_types + 1):
                for date_idx in range(n_dates):
                    # 宿泊日
                    stay_date = datetime.now() + timedelta(days=date_idx + 30)
                    
                    # 初期在庫
                    initial_stock = np.random.randint(10, 30)
                    current_stock = initial_stock
                    
                    # 各日までの予約データ
                    for days_before in range(n_days_before, -1, -1):
                        # 予約日
                        booking_date = stay_date - timedelta(days=days_before)
                        
                        # 価格（時間経過で変動）
                        base_price = 15000 + hotel_id * 2000 + plan_id * 1000
                        price_variation = 1 + 0.2 * np.sin(days_before / 10)
                        price = int(base_price * price_variation)
                        
                        # 在庫（予約が入ることで減少）
                        if days_before < n_days_before:
                            # 予約確率（宿泊日が近づくほど高くなる）
                            booking_prob = 0.1 + 0.8 * (1 - days_before / n_days_before)
                            if np.random.random() < booking_prob:
                                sold = min(current_stock, np.random.randint(1, 4))
                                current_stock -= sold
                            else:
                                sold = 0
                        else:
                            sold = 0
                        
                        data.append({
                            'hotel_id': hotel_id,
                            'plan_id': plan_id,
                            'room_type_id': room_type_id,
                            'date': stay_date.strftime('%Y-%m-%d'),
                            'created_at': booking_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'price': price,
                            'stock': current_stock
                        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ サンプルデータを作成しました: {output_path}")
    print(f"📊 データ件数: {len(df):,}件")


def main_cli():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(
        description="ホテル価格最適化システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な実行
  python src/main.py data/hotel_prices.csv
  
  # 出力ディレクトリを指定
  python src/main.py data/hotel_prices.csv --output results/
  
  # フィルタリング条件を調整
  python src/main.py data/hotel_prices.csv --min-stock 20 --min-days 10
  
  # サンプルデータを作成
  python src/main.py --create-sample data/sample_hotel_prices.csv
        """
    )
    
    parser.add_argument(
        'data_path',
        nargs='?',
        help='ホテル価格データのCSVファイルパス'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='出力ディレクトリ（デフォルト: results/）'
    )
    
    parser.add_argument(
        '--min-stock',
        type=int,
        default=10,
        help='最小初期在庫数（デフォルト: 10）'
    )
    
    parser.add_argument(
        '--min-days',
        type=int,
        default=5,
        help='最小有効日数（デフォルト: 5）'
    )
    
    parser.add_argument(
        '--create-sample',
        help='サンプルデータを作成するファイルパス'
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.create_sample)
        return
    
    if not args.data_path:
        parser.error("データファイルパスを指定してください。")
    
    if not os.path.exists(args.data_path):
        print(f"❌ ファイルが見つかりません: {args.data_path}")
        print("💡 サンプルデータを作成するには: python src/main.py --create-sample data/sample.csv")
        return
    
    main(
        data_path=args.data_path,
        output_dir=args.output,
        min_stock=args.min_stock,
        min_days=args.min_days
    )


if __name__ == "__main__":
    main_cli() 