# app.py - 改良版Streamlit UI
import requests
import streamlit as st
import random
import re

# ページ設定
st.set_page_config(
    page_title="Nemotron JP Persona Chat",
    page_icon="🤖",
    layout="wide"
)

# Google Fonts (Noto Sans JP) を読み込み
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Nemotron JP Persona Chat")
st.markdown("**Qwen3-1.7B** × **100万日本人ペルソナ** による高品質チャット")

# セッション状態の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "current_persona" not in st.session_state:
    st.session_state.current_persona = None
if "persona_index" not in st.session_state:
    st.session_state.persona_index = 0

# サイドバー：ペルソナ選択
with st.sidebar:
    st.header("🎭 ペルソナ選択")

    # ペルソナ選択方法
    persona_method = st.radio(
        "選択方法：",
        ["おすすめペルソナから選択", "ランダム選択", "番号で直接指定"]
    )

    if persona_method == "おすすめペルソナから選択":
        # 定義済みのおすすめペルソナ
        recommended_personas = {
            "東京の介護福祉士（72歳女性）": 0,
            "大阪の教師（45歳男性）": 1000,
            "札幌の看護師（30歳女性）": 2000,
            "福岡の営業（28歳男性）": 3000,
            "名古屋の主婦（55歳女性）": 4000,
            "仙台の学生（22歳女性）": 5000,
            "広島の医師（40歳男性）": 6000,
            "京都の芸術家（35歳女性）": 7000
        }

        selected_persona = st.selectbox(
            "ペルソナを選択：",
            list(recommended_personas.keys()),
            index=0
        )
        st.session_state.persona_index = recommended_personas[selected_persona]

    elif persona_method == "ランダム選択":
        if st.button("🎲 ランダムペルソナを選択", type="primary"):
            st.session_state.persona_index = random.randint(0, 999999)
        st.write(f"現在のペルソナ番号: {st.session_state.persona_index}")

    else:  # 番号で直接指定
        st.session_state.persona_index = st.number_input(
            "ペルソナ番号（0-999999）：",
            0, 999999, st.session_state.persona_index
        )

    # 現在のペルソナ情報を取得ボタン
    if st.button("📋 ペルソナ情報を取得"):
        try:
            response = requests.get(f"http://localhost:8080/personas/{st.session_state.persona_index}")
            if response.status_code == 200:
                st.session_state.current_persona = response.json()
            else:
                st.error(f"ペルソナ取得エラー: {response.status_code}")
        except Exception as e:
            st.error(f"接続エラー: {e}")

# メイン画面を2列に分割
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 チャット")

    # 会話履歴表示用のコンテナ（上部）
    chat_container = st.container()

    # 入力欄（下部）
    st.markdown("---")

    # メッセージ入力と送信ボタンを上に配置
    col_input, col_send = st.columns([4, 1])

    with col_input:
        user_input = st.text_input(
            "メッセージを入力してください：",
            placeholder="例：自己紹介をお願いします。",
            key="user_input"
        )

    with col_send:
        st.markdown("<br>", unsafe_allow_html=True)  # 高さ調整用の改行
        send_button = st.button("📤 送信", type="primary")

    # 詳細設定を下に配置
    with st.expander("⚙️ 詳細設定"):
        max_tokens = st.slider("最大トークン数", 50, 5000, 2000)
        temperature = st.slider("創造性（Temperature）", 0.0, 1.0, 0.7, 0.1)
        top_p = st.slider("語彙選択幅（Top-p）", 0.0, 1.0, 0.9, 0.1)

# チャット処理
if send_button and user_input:
    # ユーザーメッセージを履歴に追加
    st.session_state.history.append({"role": "user", "content": user_input})

    # APIリクエスト送信
    payload = {
        "messages": st.session_state.history,
        "persona_index": st.session_state.persona_index,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    try:
        with st.spinner("🤖 AI が返答を生成中..."):
            response = requests.post("http://localhost:8080/chat", json=payload, timeout=120)

        if response.status_code == 200:
            response_data = response.json()
            reply = response_data.get("reply", "エラー: レスポンスが空です")

            # <think> タグとその内容を除去
            reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)
            reply = re.sub(r'</?think>', '', reply)  # 閉じタグが残っている場合の対応
            reply = reply.strip()

            # AI応答を履歴に追加
            st.session_state.history.append({"role": "assistant", "content": reply})

            # ペルソナ情報を更新
            if response_data.get("persona_info"):
                st.session_state.current_persona = response_data["persona_info"]

        else:
            st.error(f"🚫 サーバーエラー: {response.status_code}")

    except requests.exceptions.Timeout:
        st.error("⏰ リクエストタイムアウト（120秒）。サーバーが応答しません。")
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 接続エラー: {e}")
    except Exception as e:
        st.error(f"❌ 処理エラー: {e}")

# 会話履歴表示（上部のコンテナに配置）
with chat_container:
    if st.session_state.history:
        for i, turn in enumerate(st.session_state.history):
            if turn["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(turn["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(turn["content"])
    else:
        st.info("👋 こんにちは！チャットを開始するには下のメッセージボックスに入力してください。")

# サイドバー：ペルソナ情報表示
with col2:
    st.subheader("👤 現在のペルソナ")

    if st.session_state.current_persona:
        persona = st.session_state.current_persona

        # ペルソナ情報をカード形式で表示
        st.markdown(f"""
        <div style="background-color: #000000; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #333; font-family: 'Noto Sans JP', sans-serif;">
            <h4 style="color: #ffffff; font-family: 'Noto Sans JP', sans-serif;">📋 基本情報</h4>
            <ul style="color: #ffffff; font-family: 'Noto Sans JP', sans-serif;">
                <li><strong>職業:</strong> {persona.get('occupation', '不明')}</li>
                <li><strong>年齢:</strong> {persona.get('age', '不明')}歳</li>
                <li><strong>地域:</strong> {persona.get('region', '不明')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # ペルソナ詳細
        if persona.get('persona'):
            with st.expander("📖 ペルソナ詳細", expanded=True):
                st.write(persona['persona'][:300] + ("..." if len(persona['persona']) > 300 else ""))

    else:
        st.info("🎭 「ペルソナ情報を取得」ボタンをクリックしてペルソナ詳細を表示")
        st.markdown(f"**現在のペルソナ番号:** {st.session_state.persona_index}")

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    🚀 <strong>Qwen3-1.7B</strong> × <strong>100万日本人ペルソナデータ</strong><br>
    💡 高速・高品質な日本語ペルソナチャット体験
</div>
""", unsafe_allow_html=True)

# リセットボタン
if st.button("🗑️ 会話をリセット", help="チャット履歴をクリアします"):
    st.session_state.history = []
    st.experimental_rerun()