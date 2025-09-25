# Nemo Chat App

Nemotron Personas Japan対応の日本語チャットアプリケーション

## 概要

NVIDIA Nemotron-Personas-Japanデータセット（600万ペルソナ）を活用した文化的に適切な日本語AI対応チャットアプリケーションです。

## 主な機能

- **NVIDIA-Nemotron-Nano-9B-v2**: 最新のNVIDIA Nemotronモデル対応
- **Nemotron Personas**: 100万の日本人ペルソナデータを活用
- **mamba-ssm**: 高速Mambaアーキテクチャサポート
- **Streamlit UI**: シンプルなWebインターフェース
- **FastAPI Backend**: RESTful API サーバー
- **日本語特化**: 日本の文化・社会に適応したペルソナ

## ファイル構成

```
nemo_chat_app/
├── app.py                      # Streamlit フロントエンド
├── nemotron_personas_server.py # Nemotron Personas APIサーバー
├── server.py                   # メインAPIサーバー
├── requirements.txt            # Python依存関係
└── README.md                   # このファイル
```

## 技術仕様

### 必要環境
- Python 3.10-3.12
- メモリ: 8GB以上推奨（16GB推奨）
- GPU: 24GB以上（NVIDIA-Nemotron-Nano-9B-v2推論時）
- CUDA: 12.8系対応（PyTorch 2.8.0要件）
- インターネット接続（データセット・モデル初回読み込み時）

### 依存関係
```
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
transformers==4.56.2
accelerate==1.10.1
datasets==4.1.1
mamba-ssm==2.2.5
fastapi==0.115.0
uvicorn==0.30.6
streamlit==1.38.0
sse-starlette==2.1.3
pydantic==2.9.2
```

## インストール・起動

### 1. 環境セットアップ
```bash
# 仮想環境作成・有効化
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 依存関係インストール（推奨順序）
pip install -U pip

# 1. PyTorchスタック先行インストール
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 2. AI/MLライブラリ
pip install transformers==4.56.2 accelerate==1.10.1 datasets==4.1.1

# 3. mamba-ssm（NVIDIA-Nemotron必須）
pip install mamba-ssm==2.2.5

# 4. 残りの依存関係
pip install -r requirements.txt
```

### 2. アプリケーション起動

#### FastAPI サーバー起動
```bash
# デフォルトポート8080で起動
uvicorn server:app --reload --port 8080
```

#### Streamlit UI起動
```bash
streamlit run app.py
```

## データセット詳細

### Nemotron-Personas-Japan
- **サイズ**: 100万ペルソナ（14億トークン）
- **ライセンス**: CC BY 4.0（商用利用可、クレジット要）
- **言語**: 自然な日本語
- **フィールド**: 22項目（ペルソナ6項目、コンテキスト16項目）
- **対応モデル**: NVIDIA-Nemotron-Nano-9B-v2

### 使用例
```python
from datasets import load_dataset

# データセット読み込み
personas = load_dataset("nvidia/Nemotron-Personas-Japan", split="train")

# ランダムペルソナ取得
persona = personas[0]
print(persona.keys())  # 利用可能フィールド確認
```

## 推奨モデル

### 商用利用可能
- **nvidia/NVIDIA-Nemotron-Nano-9B-v2**:
  - ライセンス: NVIDIA Open Model License（商用利用可）
  - 推論制御: `/think` または `/no_think`

### 研究用途
- **nvidia/Nemotron-H-8B-Reasoning-128K**:
  - ライセンス: 研究開発目的限定
  - 推論制御: `{'reasoning': True/False}`

## API エンドポイント

### メインAPI (port 8080)
- `GET /`: サーバー情報・ヘルスチェック
- `GET /personas/{persona_id}`: 特定ペルソナ情報取得
- `GET /stats`: ペルソナ統計情報
- `POST /chat`: チャット処理
  ```json
  {
    "messages": [{"role": "user", "content": "こんにちは"}],
    "persona_index": 0,
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.9
  }
  ```

## 高速化オプション

### vLLM使用（GPU推奨）
```bash
pip install "vllm>=0.10.1"

vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --trust-remote-code \
  --max-num-seqs 64 \
  --max-model-len 131072 \
  --mamba_ssm_cache_dtype float32
```

## トラブルシューティング

### mamba-ssm CUDA ABI互換性エラー
```bash
undefined symbol: _ZN3c1021throwNullDataPtrErrorEv
```

**解決策**：
1. **PyTorch 2.8.0必須**: mamba-ssm 2.2.5はPyTorch 2.8.0で動作
2. **インストール順序厳守**: PyTorch → transformers → mamba-ssm の順序
3. **環境クリーンアップ**:
   ```bash
   pip uninstall torch torchvision torchaudio mamba-ssm -y
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
   pip install mamba-ssm==2.2.5
   ```

### モデル読み込みエラー
- **メモリ不足**: 最低16GB RAM、24GB GPU推奨
- **CUDA不足**: CUDA 12.8系対応GPU必須
- **ネットワーク**: 初回ダウンロードで数GB必要

### パフォーマンス最適化
- **CPU推論**: `device_map="cpu"` でメモリ節約
- **Mixed Precision**: `torch_dtype=torch.bfloat16` で高速化
- **バッチ処理**: 複数リクエスト同時処理で効率向上

## ライセンス

- **コード**: MIT License
- **データセット**: CC BY 4.0 (NVIDIA Nemotron-Personas-Japan)
- **モデル**: 各モデルのライセンスに準拠

## 参考資料

- [NVIDIA Nemotron-Personas-Japan Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan)
- [Hugging Face Blog: Nemotron Personas Japan](https://huggingface.co/blog/nvidia/nemotron-personas-japan)
- [NVIDIA Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [Nemotron-H-8B-Reasoning-128K](https://huggingface.co/nvidia/Nemotron-H-8B-Reasoning-128K)