import streamlit as st
import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import db # 방금 만든 모듈

# 1. 환경 설정
load_dotenv()

def get_secret(key_name):
    try:
        if key_name in st.secrets: return st.secrets[key_name]
    except: pass
    return os.getenv(key_name)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("🚨 API 키 에러")
    st.stop()

# Pinecone 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("herb-knowledge")

GEMINI_EMBED_MODEL = "models/text-embedding-004"
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"

# --- 기능 1: 아주 단순한 검색 (오염 방지) ---
def simple_search(query_text):
    try:
        # 1. 임베딩 (질문 -> 숫자)
        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {"model": GEMINI_EMBED_MODEL, "content": {"parts": [{"text": query_text}]}}
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        # 2. Pinecone 검색 (Top 8)
        results = index.query(vector=vector, top_k=8, include_metadata=True)
        
        # 3. 텍스트로 정리
        contexts = []
        for match in results['matches']:
            m = match['metadata']
            # 검색 결과엔 체질 정보 같은 건 없고, 약초 정보만 담백하게 가져옴
            text = f"- 약초명: {m.get('name')}\n  분류: {m.get('category', '일반')}\n  효능: {m.get('efficacy')}\n  주의사항: {m.get('caution')}"
            contexts.append(text)
            
        return "\n\n".join(contexts)
    except Exception as e:
        return ""

# --- 기능 2: AI 의사 (검색결과 + 체질 짬뽕) ---
def generate_prescription(symptom, constitution, herb_list):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    # 여기서 비로소 체질과 검색결과가 만납니다.
    system_prompt = f"""
    당신은 20년 경력의 '통합 의학 전문가'입니다.
    사용자의 **체질({constitution})**을 고려하여, 검색된 약초 목록 중에서 최적의 처방을 내리세요.

    [상황]
    - 환자 체질: {constitution} (매우 중요!)
    - 호소 증상: {symptom}
    
    [검색된 약초 후보군]
    {herb_list}

    [처방 가이드]
    1. 검색된 약초 중, 환자의 증상을 치료하면서도 **체질({constitution})에 해가 되지 않는 것**을 2~3개 골라내세요.
    2. 만약 검색된 약초가 이 체질에 안 맞으면 솔직하게 경고하세요. (예: "이 약초는 차가운 성질이라 소음인에게는 추천하지 않습니다.")
    3. 약국약(Pharmacy), 한방(동의보감), 아유르베다(Ayurveda)가 섞여 있다면 적절히 조화롭게 추천하세요.
    4. 말투는 신뢰감 있고 따뜻하게 하세요.
    """
    
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": "처방을 내려주세요."}]}]
    }
    
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text']
    except:
        return "죄송합니다. 처방을 생성하는 중 문제가 발생했습니다."

# --- 메인 화면 UI ---
st.set_page_config(page_title="체질 맞춤 약초원", page_icon="🌿")

st.title("🌿 체질 맞춤 약초 처방")

# 1. 로그인 (ID 입력)
if "user_id" not in st.session_state:
    with st.form("login_form"):
        st.subheader("📋 진료 접수")
        input_id = st.text_input("성함 또는 ID를 입력하세요")
        if st.form_submit_button("진료 시작"):
            if input_id:
                st.session_state.user_id = input_id
                st.rerun() # 새로고침해서 다음 단계로
    st.stop() # ID 없으면 여기서 멈춤

# 2. 체질 확인 (DB 조회 -> 없으면 등록 -> 있으면 통과)
if "constitution" not in st.session_state:
    # DB에서 찾아봄
    saved_const = db.get_user_constitution(st.session_state.user_id)
    
    if saved_const:
        st.session_state.constitution = saved_const
        st.toast(f"환영합니다! {st.session_state.user_id}님 ({saved_const})", icon="✅")
    else:
        # DB에 없으면 물어봄 (최초 1회)
        st.info(f"반갑습니다 {st.session_state.user_id}님, 처음 오셨군요!")
        with st.form("const_form"):
            st.write("정확한 처방을 위해 **체질**을 한 번만 알려주세요.")
            selected = st.selectbox("나의 체질은?", ["소음인", "소양인", "태음인", "태양인", "잘 모름(아유르베다 바타/피타/카파)"])
            
            if st.form_submit_button("정보 저장"):
                db.register_user(st.session_state.user_id, selected)
                st.session_state.constitution = selected
                st.rerun()
        st.stop()

# 3. 진료실 (채팅)
st.subheader(f"👨‍⚕️ {st.session_state.constitution}인 {st.session_state.user_id}님, 어디가 불편하세요?")

# 채팅 기록 표시
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "증상을 편하게 말씀해주세요. (예: 소화가 안 되고 머리가 아파요)"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 입력 처리
if prompt := st.chat_input("증상을 입력하세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        # 1단계: 단순 검색 (증상 -> 약초 리스트)
        with st.spinner("약초 서랍을 뒤지는 중..."):
            herbs_found = simple_search(prompt)
        
        # 2단계: 맞춤 처방 (약초 리스트 + 체질 -> 최종 처방)
        with st.spinner(f"{st.session_state.constitution} 체질에 맞춰 분석 중..."):
            diagnosis = generate_prescription(
                symptom=prompt,
                constitution=st.session_state.constitution,
                herb_list=herbs_found
            )
            
            st.markdown(diagnosis)
            st.session_state.messages.append({"role": "assistant", "content": diagnosis})
            
            # 기록 저장
            db.save_diagnosis(
                st.session_state.user_id, 
                prompt, 
                "AI 진단 완료", 
                diagnosis[:100] # 엑셀엔 너무 기니까 100자만 저장
            )
