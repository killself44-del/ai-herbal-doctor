import streamlit as st
import os
import time
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # 위에서 만든 db.py 임포트

# 환경 변수 및 설정
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("herb-knowledge")
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# --- 핵심 함수 ---

def simple_search(query_text):
    """Pinecone 검색 및 디버그 정보 반환"""
    try:
        # 검색어 보정 (아유르베다/약국약 가중치)
        search_query = f"증상 '{query_text}'에 대한 약국 약, 한방 약초, 아유르베다 통합 처방"
        if any(keyword in query_text for keyword in ["인도", "아유르베다", "Ayurveda"]):
            search_query = f"Ayurveda 인도 아유르베다 허브 처방: {query_text}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {"model": GEMINI_EMBED_MODEL, "content": {"parts": [{"text": search_query}]}, "taskType": "RETRIEVAL_QUERY"}
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        # 넉넉하게 20개를 뽑아 순위권 밖 약물 방지
        results = index.query(vector=vector, top_k=20, include_metadata=True)
        
        contexts, debug_list = [], []
        for match in results['matches']:
            m = match['metadata']
            contexts.append(f"[{m.get('category')}] {m.get('name')}: {m.get('efficacy')}")
            debug_list.append(f"[{match['score']:.2f}] {m.get('name')} ({m.get('category')})")
        return "\n\n".join(contexts), debug_list
    except:
        return "데이터 검색 실패", []

def generate_prescription(symptom, constitution, herb_list):
    """Gemini 처방 생성"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    system_prompt = f"""
    당신은 통합 의학 전문가입니다. 체질({constitution})과 증상({symptom})을 바탕으로 처방하세요.
    1. 빠른 효과는 [Pharmacy](약국약) 우선. 
    2. 근본 치료는 [동의보감] 또는 [Ayurveda] 추천.
    3. 소음인 체질 특성을 반드시 반영하여 주의사항을 넣으세요.
    검색 데이터: {herb_list}
    """
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text']
    except:
        return "처방 생성 오류"

# --- 화면 구성 ---

st.set_page_config(page_title="통합 AI 약국", page_icon="🌿", layout="wide")

# 1. 로그인
if "user_id" not in st.session_state:
    st.title("🌿 통합 AI 체질 약국")
    with st.form("login"):
        uid = st.text_input("아이디(이름) 입력")
        if st.form_submit_button("입장"):
            st.session_state.user_id = uid
            st.rerun()
    st.stop()

# 2. 체질 확인 (DB 연동 및 퀴즈)
if "constitution" not in st.session_state:
    saved_c = db.get_user_constitution(st.session_state.user_id)
    if saved_c:
        st.session_state.constitution = saved_c
    else:
        st.subheader(f"🔍 {st.session_state.user_id}님, 체질 진단을 시작합니다.")
        with st.form("quiz"):
            q1 = st.radio("소화 상태", ["자주 체한다", "보통", "매우 좋다"])
            q2 = st.radio("온도 민감도", ["추위를 탄다", "보통", "열이 많다"])
            if st.form_submit_button("진단 완료"):
                res = "소음인" if "추위" in q2 or "체한다" in q1 else ("소양인" if "열" in q2 else "태음인")
                st.session_state.constitution = res
                db.save_user_constitution(st.session_state.user_id, res)
                st.rerun()
        st.stop()

# 3. 진료실
st.header(f"👨‍⚕️ {st.session_state.user_id}님 ({st.session_state.constitution}) 진료실")

if prompt := st.chat_input("증상을 입력하세요"):
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            context, debug = simple_search(prompt)
            with st.sidebar:
                st.write("🔍 실시간 검색 결과")
                for d in debug: st.caption(d)
            
            ans = generate_prescription(prompt, st.session_state.constitution, context)
            st.markdown(ans)
            db.save_diagnosis(st.session_state.user_id, prompt, "통합처방", ans[:50])
