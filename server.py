# =============================================================================
# ライブラリインポート
# =============================================================================
from typing import List, Optional  # 型ヒント用（ListとOptionalを使用）
from fastapi import FastAPI, HTTPException  # WebAPIフレームワーク
from pydantic import BaseModel  # データ検証・シリアライゼーション
import torch  # PyTorch（深層学習フレームワーク）
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face transformers
from datasets import load_dataset  # データセット読み込み用
import logging  # ログ出力用

# =============================================================================
# ロギング設定
# =============================================================================
logging.basicConfig(level=logging.INFO)  # INFO レベル以上のログを出力
logger = logging.getLogger(__name__)  # このモジュール専用のロガーを取得

# =============================================================================
# FastAPIアプリケーション初期化
# =============================================================================
app = FastAPI(title="Nemotron JP Persona Chat", version="2.0.0")

# =============================================================================
# グローバル変数（アプリケーション全体で使用する変数）
# =============================================================================
MODEL_ID = "Qwen/Qwen3-1.7B"  # 使用する言語モデルのID（Hugging Faceから取得）
tokenizer = None  # テキストをトークン（数値）に変換するツール
model = None  # 実際の言語生成モデル
personas = None  # 日本人ペルソナデータセット
startup_error = None  # サーバー起動時のエラーを記録する変数

# =============================================================================
# データモデル定義（APIの入力・出力の形式を定義）
# =============================================================================

class ChatTurn(BaseModel):
    """
    チャットの1回のやりとりを表すクラス
    - role: 'user'（ユーザー）または 'assistant'（AI）
    - content: 実際の発言内容
    """
    role: str  # 発言者の役割
    content: str  # 発言内容

class ChatRequest(BaseModel):
    """
    チャットAPIへのリクエスト形式
    フロントエンドからこの形式でデータを受け取る
    """
    messages: List[ChatTurn]  # これまでの会話履歴のリスト
    persona_index: Optional[int] = 0  # 使用するペルソナのインデックス（0から開始）
    max_new_tokens: int = 150  # 生成する最大トークン数（長さの制限）
    temperature: float = 0.8  # 生成のランダム性（0.0-1.0、高いほど創造的）
    top_p: float = 0.9  # 生成時の語彙選択幅（0.0-1.0、nucleus sampling）

class ChatResponse(BaseModel):
    """
    チャットAPIからのレスポンス形式
    この形式でフロントエンドにデータを返す
    """
    reply: str  # AIの返答内容
    persona_info: Optional[dict] = None  # 使用したペルソナの情報（optional）

# =============================================================================
# サーバー起動時の初期化処理
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    FastAPIサーバー起動時に実行される関数
    モデルとデータセットの読み込みを行う
    """
    global tokenizer, model, personas, startup_error

    try:
        # ステップ1: 日本人ペルソナデータセットの読み込み
        logger.info("Loading Nemotron-Personas-Japan dataset...")
        personas = load_dataset("nvidia/Nemotron-Personas-Japan", split="train")
        logger.info(f"Loaded {len(personas)} Japanese personas")

        # ステップ2: Qwen3-1.7Bモデルの読み込み
        logger.info("Loading Qwen3-1.7B model...")

        # トークナイザー（文章を数値に変換するツール）の読み込み
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        # パディングトークンが設定されていない場合、終了トークンを使用
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 言語モデル本体の読み込み
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # メモリ効率のため16bit浮動小数点を使用
            device_map="auto",  # 利用可能なGPU/CPUに自動でモデルを配置
            trust_remote_code=True  # Hugging Faceのカスタムコード実行を許可
        )

        # ステップ3: 初期化成功
        startup_error = None  # 成功時はエラーをクリア
        logger.info("System ready for persona-aware chat with Qwen3-1.7B")

    except Exception as e:
        # 初期化に失敗した場合のエラーハンドリング
        error_msg = f"Startup error: {e}"
        logger.error(error_msg)
        startup_error = error_msg  # エラー状態を記録

# =============================================================================
# APIエンドポイント定義
# =============================================================================

@app.get("/")
async def root():
    """
    ルートエンドポイント（サーバーの基本情報を返す）
    ブラウザで http://localhost:8080/ にアクセスした時に表示される情報
    """
    return {
        "message": "Nemotron JP Persona Chat Server v2.0 with Qwen3-1.7B",
        "status": "error" if startup_error else "running",  # サーバーの状態
        "model": MODEL_ID,  # 使用中のモデル名
        "model_loaded": model is not None,  # モデルが正常に読み込まれているか
        "personas_loaded": personas is not None,  # ペルソナが読み込まれているか
        "total_personas": len(personas) if personas else 0,  # ペルソナの総数
        "startup_error": startup_error  # 起動時エラーがあれば表示
    }

@app.get("/personas/{persona_id}")
async def get_persona(persona_id: int):
    """
    特定のペルソナ情報を取得するエンドポイント
    例: GET /personas/0 で0番目のペルソナ情報を取得

    Args:
        persona_id (int): 取得したいペルソナのID（0から開始）

    Returns:
        dict: ペルソナの詳細情報（職業、年齢、地域、性格など）
    """
    # ペルソナデータが読み込まれているかチェック
    if personas is None:
        raise HTTPException(status_code=503, detail="Personas not loaded")

    # 指定されたIDが存在するかチェック
    if persona_id >= len(personas):
        raise HTTPException(status_code=404, detail="Persona not found")

    # ペルソナ情報を返す
    return personas[persona_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    メインのチャットエンドポイント
    ユーザーのメッセージを受け取り、指定されたペルソナでAIが返答する

    Args:
        req (ChatRequest): チャットリクエスト（会話履歴、ペルソナID、生成パラメータなど）

    Returns:
        ChatResponse: AIの返答とペルソナ情報
    """
    try:
        # =================================================================
        # ステップ1: システム状態とリクエストの妥当性チェック
        # =================================================================

        # サーバー起動時にエラーが発生していないかチェック
        if startup_error:
            raise HTTPException(status_code=503, detail=f"System not ready: {startup_error}")

        # ペルソナデータが存在し、指定されたペルソナIDが有効かチェック
        if not personas or req.persona_index >= len(personas):
            raise HTTPException(status_code=400, detail="Invalid persona index")

        # 指定されたペルソナのデータを取得
        persona_data = personas[req.persona_index]

        # =================================================================
        # ステップ2: モデル未読み込み時のフォールバック処理
        # =================================================================

        # モデルまたはトークナイザーが読み込まれていない場合の簡易応答
        if model is None or tokenizer is None:
            persona_text = persona_data.get("persona", "日本人です")
            reply = f"{persona_text[:100]}... こんにちは！何かお手伝いできることはありますか？"
            return ChatResponse(
                reply=reply,
                persona_info={"persona": persona_text[:200]}
            )

        # ペルソナ情報の抽出

        # ペルソナデータから各項目を取得（存在しない場合は空文字）
        persona_description = persona_data.get("persona", "")  # 人物の詳細説明
        occupation = persona_data.get("occupation", "")  # 職業
        age = persona_data.get("age", "")  # 年齢
        region = persona_data.get("region", "")  # 出身・居住地

        # ペルソナプロンプトの構築

        # 存在するペルソナ情報のみをリストに追加
        persona_parts = []
        if persona_description:
            persona_parts.append(f"人物像：{persona_description}")
        if occupation:
            persona_parts.append(f"職業：{occupation}")
        if age:
            persona_parts.append(f"年齢：{age}歳")
        if region:
            persona_parts.append(f"出身・居住地：{region}")

        # ペルソナ情報を改行で結合（情報がない場合はデフォルト）
        persona_info_text = "\n".join(persona_parts) if persona_parts else "一般的な日本人"

        # AIに与えるシステムプロンプトを作成
        persona_prompt = f"""あなたは以下のペルソナの人物として、その人になりきって自然な日本語で返答してください。メタ的な説明や分析は一切せず、その人物そのものとして話してください。

【あなたの人物像】
{persona_info_text}

この人物として、自然な口調と視点で会話してください。その人の経験、価値観、話し方で返答してください。"""

        # 会話履歴の構築

        # メッセージリストを初期化し、まずシステムプロンプトを追加
        messages = []
        messages.append({"role": "system", "content": persona_prompt})

        # リクエストから過去の会話履歴を追加
        # これによりAIが会話の文脈を理解し、ペルソナを一貫して維持できる
        for msg in req.messages:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append({"role": "assistant", "content": msg.content})

        # Qwen3専用フォーマットへの変換

        # Qwen3が理解できる特殊トークン形式に変換
        # <|im_start|>と<|im_end|>はQwen3の会話形式で必要な区切り文字
        conversation = ""
        for msg in messages:
            if msg["role"] == "system":
                conversation += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                conversation += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                conversation += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

        # AIの返答を促すための開始タグを追加
        conversation += "<|im_start|>assistant\n"

        # テキストのトークン化

        # 会話テキストをモデルが理解できる数値（トークン）に変換
        inputs = tokenizer.encode(
            conversation,
            return_tensors="pt",  # PyTorchテンソル形式で返す
            max_length=1024,  # 最大1024トークンまで（Qwen3-1.7Bに合わせて調整）
            truncation=True  # 長すぎる場合は切り詰める
        ).to(model.device)  # モデルと同じデバイス（GPU/CPU）に配置

        # AIによるテキスト生成

        # 推論モード（勾配計算なし）でメモリ効率よく生成
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=req.max_new_tokens,  # 新しく生成するトークン数の上限
                do_sample=True,  # ランダムサンプリングを有効にする
                temperature=req.temperature,  # 生成の創造性（0.0-1.0）
                top_p=req.top_p,  # nucleus sampling（語彙選択幅）
                pad_token_id=tokenizer.pad_token_id,  # パディング用トークン
                eos_token_id=tokenizer.eos_token_id,  # 終了トークン
                repetition_penalty=1.1,  # 同じ表現の繰り返しを軽減
                no_repeat_ngram_size=2  # 2語の組み合わせの繰り返しを防止
            )

        # 生成結果のデコード

        # 生成されたトークンを人間が読める文字列に変換
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


        # レスポンスの後処理

        # まず<think>タグとその内容を除去
        import re
        cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'</?think[^>]*>', '', cleaned_text)

        # assistant以降の部分のみを抽出
        if "assistant" in cleaned_text:
            # 最後のassistantの位置を探す
            assistant_parts = cleaned_text.split("assistant")
            if len(assistant_parts) > 1:
                # 最後のassistant以降の部分を取得
                reply = assistant_parts[-1].strip()
            else:
                reply = cleaned_text
        else:
            reply = cleaned_text


        # レスポンス情報の構築

        # フロントエンドに返すペルソナ情報を整理
        persona_info = {
            "persona": persona_description[:200],  # 説明文は200文字以内に制限
            "occupation": occupation,  # 職業
            "age": age,  # 年齢
            "region": region  # 出身・居住地
        }

        # ChatResponse形式でレスポンスを返す
        return ChatResponse(
            reply=reply,  # AIの返答
            persona_info=persona_info  # ペルソナ詳細
        )

    except Exception as e:
        # エラーが発生した場合のハンドリング
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    サーバーの健康状態をチェックするエンドポイント
    外部監視システムやロードバランサーから定期的に呼び出される

    Returns:
        dict: サーバーの状態情報（healthy/unhealthy, モデル読み込み状況など）
    """
    return {
        "status": "unhealthy" if startup_error else "healthy",  # 全体的な健康状態
        "model": MODEL_ID,  # 使用モデル名
        "model_loaded": model is not None,  # モデルが読み込まれているか
        "personas_loaded": personas is not None,  # ペルソナが読み込まれているか
        "total_personas": len(personas) if personas else 0,  # ペルソナ総数
        "startup_error": startup_error  # 起動エラーの詳細（あれば）
    }

@app.get("/stats")
async def get_stats():
    """
    ペルソナデータセットの統計情報を取得するエンドポイント
    データセット内の職業分布、地域分布、年齢分布などを返す

    Returns:
        dict: 統計情報（職業TOP10、地域分布、年齢層分布など）
    """
    # ペルソナデータが読み込まれているかチェック
    if not personas:
        raise HTTPException(status_code=503, detail="Personas not loaded")

    # 統計用の辞書を初期化
    occupations = {}  # 職業別カウント
    regions = {}      # 地域別カウント
    ages = {}         # 年齢層別カウント

    # 全ペルソナを処理して統計を作成
    for persona in personas:
        # 職業情報を取得・集計
        occ = persona.get("occupation", "不明")
        occupations[occ] = occupations.get(occ, 0) + 1

        # 地域情報を取得・集計
        reg = persona.get("region", "不明")
        regions[reg] = regions.get(reg, 0) + 1

        # 年齢情報を取得・年代別に集計
        age = persona.get("age")
        if age:
            age_group = f"{age//10*10}代"  # 20代、30代など
            ages[age_group] = ages.get(age_group, 0) + 1

    return {
        "total_personas": len(personas),  # 総ペルソナ数
        "top_occupations": sorted(occupations.items(), key=lambda x: x[1], reverse=True)[:10],  # 職業TOP10
        "regions": dict(sorted(regions.items(), key=lambda x: x[1], reverse=True)),  # 地域分布（多い順）
        "age_groups": dict(sorted(ages.items()))  # 年齢層分布（年代順）
    }

# =============================================================================
# メイン実行部分
# =============================================================================

if __name__ == "__main__":
    """
    スクリプトが直接実行された場合のメイン処理
    python server.py で実行された時に動作する
    """
    import uvicorn

    # UvicornでFastAPIサーバーを起動
    # host="0.0.0.0" : すべてのネットワークインターフェースでリッスン
    # port=8080 : ポート8080でサーバーを起動
    uvicorn.run(app, host="0.0.0.0", port=8080)