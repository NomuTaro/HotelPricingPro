# Hotel Pricing Pro - ホテル価格最適化システム

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 概要 / Overview

ホテルの売上最大化を目指し、過去の予約データに基づいて日ごとの価格を動的に最適化する機械学習システムです。

This is a machine learning system that optimizes daily hotel pricing dynamically based on historical booking data to maximize revenue.

## 主な機能 / Key Features

- **データ前処理**: 予約データの読み込みと前処理
- **機械学習モデル**: LightGBMを使用した予約数予測モデル
- **価格最適化**: 理想のブッキングカーブに基づく動的価格設定
- **可視化**: 実績データとシミュレーション結果の比較グラフ
- **分析機能**: 定量的・定性的なブッキングカーブ分析

## 技術スタック / Tech Stack

- **Python 3.8+**
- **Pandas**: データ処理
- **NumPy**: 数値計算
- **LightGBM**: 機械学習モデル
- **Scikit-learn**: モデル評価
- **Matplotlib**: 可視化
- **SciPy**: 最適化・統計分析

## インストール / Installation

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/HotelPricingPro.git
cd HotelPricingPro

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

## 使用方法 / Usage

### 1. データ準備
```python
# データファイルを配置
# hotel_prices.csv を data/ ディレクトリに配置
```

### 2. 基本的な実行
```python
# Jupyter Notebookを起動
jupyter notebook

# Research01.ipynb を開いて実行
```

### 3. コマンドライン実行
```bash
python src/main.py --data_path data/hotel_prices.csv
```

## プロジェクト構造 / Project Structure

```
HotelPricingPro/
├── README.md                 # プロジェクト説明
├── requirements.txt          # 依存関係
├── .gitignore               # Git除外設定
├── LICENSE                  # ライセンス
├── data/                    # データファイル
│   ├── hotel_prices.csv     # ホテル価格データ
│   └── events.csv          # イベントデータ（オプション）
├── src/                     # ソースコード
│   ├── __init__.py
│   ├── main.py             # メイン実行ファイル
│   ├── data_processor.py   # データ処理クラス
│   ├── booking_analyzer.py # ブッキング分析クラス
│   ├── price_optimizer.py  # 価格最適化クラス
│   └── visualizer.py       # 可視化クラス
├── notebooks/              # Jupyter Notebooks
│   └── Research01.ipynb    # 研究用ノートブック
├── tests/                  # テストファイル
│   └── test_*.py
├── docs/                   # ドキュメント
│   └── api.md
└── results/               # 結果出力
    └── figures/
```

## データ形式 / Data Format

### hotel_prices.csv
```csv
hotel_id,plan_id,room_type_id,date,created_at,price,stock
1,1,1,2025-01-15,2024-10-15,15000,10
1,1,1,2025-01-15,2024-10-16,15500,9
...
```

### events.csv (オプション)
```csv
event_date,event_name,event_type
2025-01-15,大阪マラソン,sports
2025-02-14,バレンタインデー,holiday
...
```

## アルゴリズム / Algorithm

### 1. ブッキングカーブ分析
- 指数関数による予約率のモデル化
- α（予約強度）パラメータの算出

### 2. 機械学習モデル
- LightGBMによる日次予約数予測
- 特徴量: 価格、曜日、残り日数、現在の予約率、在庫数

### 3. 価格最適化
- 理想のブッキングカーブに基づく動的価格設定
- RevPAR（部屋単価収益）の最大化

## 開発環境セットアップ / Development Setup

### 1. 依存関係の管理
```bash
# 開発用依存関係も含めてインストール
pip install -r requirements-dev.txt

# コードフォーマット
black src/
isort src/

# リント
flake8 src/
pylint src/
```

### 2. テスト実行
```bash
# 全テスト実行
pytest tests/

# カバレッジ付きテスト
pytest --cov=src tests/
```

### 3. ドキュメント生成
```bash
# Sphinxドキュメント生成
cd docs
make html
```

## 貢献 / Contributing

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス / License

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 連絡先 / Contact

- プロジェクト管理者: [Your Name]
- メール: [your.email@example.com]
- プロジェクトリンク: [https://github.com/yourusername/HotelPricingPro](https://github.com/yourusername/HotelPricingPro)

## 謝辞 / Acknowledgments

- ホテルデータ提供: [データ提供元]
- 研究協力: [協力者名]
- 参考論文: Shintani & Umeno (2023) - "Average booking curves draw exponential functions" 