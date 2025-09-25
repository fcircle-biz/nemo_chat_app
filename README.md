# 🤖 Nemotron JP Persona Chat

**Qwen3-1.7B** × **100万日本人ペルソナ** による高品質日本語チャットアプリケーション

## 概要

NVIDIA Nemotron-Personas-Japanデータセット（100万ペルソナ）とQwen3-1.7Bモデルを活用した、文化的に適切な日本語AI対応チャットアプリケーションです。

## 主な機能

- **🧠 Qwen3-1.7B**: 高性能で軽量な中国製言語モデル
- **👤 Nemotron Personas**: 100万の日本人ペルソナデータを活用
- **🎨 改良されたUI**: 底部入力レイアウトによる直感的なチャット体験
- **⚡ Streamlit Frontend**: レスポンシブなWebインターフェース
- **🚀 FastAPI Backend**: 高性能RESTful APIサーバー
- **🎭 ペルソナ選択**: おすすめ・ランダム・直接指定による柔軟な選択
- **⚙️ 詳細設定**: トークン数・温度・Top-pの調整可能

## ファイル構成

```
nemo_chat_app/
├── app.py           # Streamlit フロントエンド（底部入力レイアウト）
├── server.py        # FastAPI バックエンドサーバー
├── requirements.txt # Python依存関係
├── .gitignore       # Git除外設定
└── README.md        # このファイル
```

## 技術仕様

### 必要環境
- **Python**: 3.10-3.12
- **メモリ**: 8GB以上（16GB推奨）
- **GPU**: 4GB以上（Qwen3-1.7B推論時、CPU推論も対応）
- **CUDA**: 12.8系対応（PyTorch 2.8.0要件）
- **ストレージ**: 10GB以上（モデル・データセット含む）
- **インターネット**: 初回セットアップ時

### 主要依存関係
```
torch==2.8.0
transformers==4.56.2
accelerate==1.10.1
datasets==4.1.1
fastapi==0.115.0
uvicorn==0.30.6
streamlit==1.38.0
requests==2.32.3
```

## インストール・起動

### 1. 環境セットアップ
```bash
# リポジトリクローン
git clone <repository-url>
cd nemo_chat_app

# 仮想環境作成・有効化
python -m venv .venv
source .venv/bin/activate  # Linux/Mac (.venv\Scripts\activate on Windows)

# 依存関係インストール
pip install -U pip
pip install -r requirements.txt
```

### 2. アプリケーション起動

#### 1. FastAPI サーバー起動（ターミナル1）
```bash
source .venv/bin/activate
python server.py
```
サーバーは http://localhost:8080 で起動します

#### 2. Streamlit UI起動（ターミナル2）
```bash
source .venv/bin/activate
streamlit run app.py
```
UIは http://localhost:8501 で起動します

## UI機能

### チャットインターフェース
- **💬 底部入力レイアウト**: 一般的なチャットアプリと同様の直感的な配置
- **📜 会話履歴**: 上部に会話履歴を表示、自動スクロール対応
- **⚙️ 詳細設定**: 最大トークン数、Temperature、Top-p調整
- **🎭 ペルソナ選択**:
  - おすすめペルソナから選択
  - ランダム選択
  - 番号による直接指定（0-999999）

### ペルソナデータ
- **データセット**: NVIDIA Nemotron-Personas-Japan
- **サイズ**: 100万ペルソナ
- **ライセンス**: CC BY 4.0
- **内容**: 日本の多様な職業・年齢・地域のペルソナ
- **主要フィールド**: occupation, age, region, persona（詳細プロフィール）

## 使用モデル

- **Qwen3-1.7B**:
  - 高性能で軽量な言語モデル
  - CPU/GPU両対応
  - `<think>`タグフィルタリング機能

## API エンドポイント

### FastAPI サーバー (port 8080)
- **GET** `/`: サーバーヘルスチェック
- **GET** `/personas/{persona_id}`: ペルソナ情報取得
- **POST** `/chat`: チャット処理

```json
POST /chat
{
  "messages": [{"role": "user", "content": "こんにちは"}],
  "persona_index": 0,
  "max_new_tokens": 2000,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## 特徴

### 🎯 主な改良点
- **底部入力レイアウト**: 直感的なチャット体験
- **レスポンス処理**: `<think>`タグの自動除去
- **ペルソナ統合**: シームレスなペルソナ切り替え
- **日本語フォント**: Noto Sans JPによる美しい表示

### 🔧 技術的特徴
- **軽量**: Qwen3-1.7Bによる高速推論
- **柔軟**: CPU/GPU両対応
- **スケーラブル**: FastAPI + Streamlitアーキテクチャ
- **オープンソース**: MITライセンス

## ライセンス

- **コード**: MIT License
- **データセット**: CC BY 4.0 (NVIDIA Nemotron-Personas-Japan)
- **モデル**: Qwen3ライセンスに準拠

## 参考リンク

- [NVIDIA Nemotron-Personas-Japan Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan)
- [Qwen3-1.7B Model](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)