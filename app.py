import streamlit as st
import os
import time
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import db # 구글 시트 모듈

load_dotenv()

# API 키 로드
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("herb-knowledge")
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

def simple_search(query_text):
    try:
        # 검색어 확장: '두통'뿐만 아니라 '치료', '약국' 등을 섞어 검색 품질 향상
        enhanced_query = f"증상 '{query_text}'에 효과적인 약국 약과 한방 약초 처방"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {"model": GEMINI_EMBED_MODEL, "content": {"parts": [{"text": enhanced_query}]}, "taskType": "RETRIEVAL_QUERY"}
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        # 🌟 검색 개수를 20개로 늘려 약국 약이 밀려나지 않게 함
        results = index.query(vector=vector, top_k=20, include_metadata=True)
        
        contexts = []
        debug_list = []
        for match in results['matches']:
            m = match['metadata']
            text = f"- [{m.get('category')}] {m.get('name')}: {m.get('efficacy')}"
            contexts.append(text)
            debug_list.append(f"[{match['score']:.2f}] {m.get('name')} ({m.get('category')})")
        return "\n\n".join(contexts), debug_list
    except:
        return "", []

def generate_prescription(symptom, constitution, herb_list):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    system_prompt = f"""
    당신은 통합 의학 전문가입니다. 환자의 체질({constitution})과 증상({symptom})을 분석하세요.
    
    [처방 원칙]
    1. 급성 통증(두통 등)에는 검색된 [Pharmacy] 약국 약을 우선 추천하세요.
    2. 체질 개선과 근본 치료에는 [동의보감] 또는 [Ayurveda]를 활용하세요.
    3. 소음인 체질인 경우 차가운 성질의 약초(괴화 등)는 신중하게 처방하세요.
    
    [검색 데이터]
    {herb_list}
    """
    
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text']
    except:
        return "처방 생성 실패"

# --- UI 레이아웃 ---
st.set_page_config(page_title="통합 AI 약국", page_icon="🌿")
st.title("🌿 통합 AI 체질 맞춤 약국")

# 로그인 및 체질 진단 로직 (기존과 동일하되 세션 관리 강화)
if "user_id" not in st.session_state:
    with st.form("login"):
        uid = st.text_input("아이디 입력")
        if st.form_submit_button("입장"):
            st.session_state.user_id = uid
            st.rerun()
    st.stop()

# 체질 확인 (DB 연동)
if "constitution" not in st.session_state:
    with st.spinner("회원 정보 확인 중..."):
        saved_c = db.get_user_constitution(st.session_state.user_id)
        if saved_c:
            st.session_state.constitution = saved_c
        else:
            # 신규 회원이면 진단 퀴즈 (생략 - 이전 코드 참조)
            st.session_state.constitution = "소음인" # 임시

# 진료실 UI
st.subheader(f"👨‍⚕️ {st.session_state.user_id}님 ({st.session_state.constitution}) 진료실")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("어디가 아프신가요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("최적의 처방을 찾는 중..."):
            herbs_text, debug_info = simple_search(prompt)
            
            # 사이드바 디버깅
            with st.sidebar:
                st.write("🔍 검색된 후보 (Top 20)")
                for d in debug_info: st.caption(d)

            ans = generate_prescription(prompt, st.session_state.constitution, herbs_text)
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            
            # 🌟 구글 시트 저장 (에러 방지 처리)
            try:
                db.save_diagnosis(st.session_state.user_id, prompt, "통합 처방", ans[:50])
            except:
                pass
