import streamlit as st
import os
import time  # 👈 이 친구가 빠져서 에러가 났습니다. 추가 완료!
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # 구글 시트 모듈

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
    st.error("🚨 API 키 에러: .env 파일이나 Secrets를 확인하세요.")
    st.stop()

# Pinecone & Model 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("herb-knowledge")
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# --- 기능 1: AI 체질 판별사 (신규 가입용) ---
def analyze_constitution(answers):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    # 답변을 모아서 프롬프트 생성
    user_data = f"""
    1. 체격: {answers['body']}
    2. 소화: {answers['digestion']}
    3. 추위/더위: {answers['temp']}
    4. 땀: {answers['sweat']}
    5. 성격: {answers['mind']}
    """
    
    system_prompt = f"""
    당신은 사상체질 진단 전문가입니다. 사용자의 답변을 분석해 '태양인', '태음인', '소양인', '소음인' 중 하나로 결론 내리세요.
    
    [사용자 답변]
    {user_data}
    
    [분석 규칙]
    - 소음인: 소화 기능 약함, 추위 탐, 꼼꼼함.
    - 소양인: 소화 잘됨, 열 많음, 급함.
    - 태음인: 골격 큼, 땀 많음, 잘 먹음.
    - 태양인: 매우 드뭄, 폐 기능 강함, 독창적.
    
    [출력 형식]
    설명은 생략하고 오직 **체질명 단어 하나만** 출력하세요. (예: 소음인)
    """
    
    payload = {
        "contents": [{"parts": [{"text": system_prompt}]}]
    }
    
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text'].strip()
    except:
        return "알 수 없음"

# --- 기능 2: 단순 검색 (증상 -> 약초 리스트) ---
def simple_search(query_text):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {"model": GEMINI_EMBED_MODEL, "content": {"parts": [{"text": query_text}]}}
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        results = index.query(vector=vector, top_k=8, include_metadata=True)
        
        contexts = []
        for match in results['matches']:
            m = match['metadata']
            text = f"- 약초명: {m.get('name')}\n  분류: {m.get('category')}\n  효능: {m.get('efficacy')}\n  주의사항: {m.get('caution')}"
            contexts.append(text)
        return "\n\n".join(contexts)
    except:
        return ""

# --- 기능 3: 맞춤 처방 (체질 + 검색결과) ---
def generate_prescription(symptom, constitution, herb_list):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    system_prompt = f"""
    당신은 통합 의학 전문가입니다.
    환자의 **체질({constitution})**을 고려하여, 아래 검색된 약초들 중 가장 적합한 것을 처방하세요.
    
    [환자 정보]
    - 체질: {constitution}
    - 증상: {symptom}
    
    [검색된 약초 목록]
    {herb_list}
    
    [지침]
    1. 이 체질에 가장 잘 맞는 약초를 1순위로 추천하세요.
    2. 체질에 맞지 않는 약초는 경고하거나 제외하세요.
    3. 한방, 아유르베다, 약국약을 골고루 고려하세요.
    """
    
    payload = {
        "contents": [{"parts": [{"text": system_prompt}]}]
    }
    
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text']
    except:
        return "처방 생성 실패"

# --- 메인 앱 UI ---
st.set_page_config(page_title="체질 맞춤 약초원", page_icon="🌿")
st.title("🌿 AI 체질 맞춤 처방소")

# 1. 로그인
if "user_id" not in st.session_state:
    with st.form("login_form"):
        st.subheader("📋 진료 접수")
        input_id = st.text_input("성함 또는 ID")
        if st.form_submit_button("입장"):
            if input_id:
                st.session_state.user_id = input_id
                st.rerun()
    st.stop()

# 2. 체질 확인 (없으면 -> 진단 테스트 실행)
if "constitution" not in st.session_state:
    # DB 조회
    saved_const = db.get_user_constitution(st.session_state.user_id)
    
    if saved_const:
        st.session_state.constitution = saved_const
        st.toast(f"환영합니다! {saved_const} 체질의 {st.session_state.user_id}님", icon="✅")
    else:
        # 🌟 체질 진단 퀴즈 🌟
        st.info(f"반갑습니다 {st.session_state.user_id}님! 정확한 처방을 위해 체질 진단을 먼저 진행합니다.")
        
        with st.form("quiz_form"):
            st.markdown("### 🕵️‍♂️ 30초 체질 진단 테스트")
            
            q1 = st.radio("1. 평소 체격이나 체형은 어떤가요?", 
                ["상체가 발달하고 어깨가 넓다", "하체가 발달하고 골반이 넓다", "전체적으로 통통하고 골격이 크다", "전체적으로 마르고 약해 보인다"])
            
            q2 = st.radio("2. 소화 기능은 어떤가요?", 
                ["아주 잘 먹고 소화도 빠르다", "소화가 자주 안 되고 입이 짧다", "폭식하는 경향이 있고 살이 잘 찐다", "평범하다"])
            
            q3 = st.radio("3. 추위와 더위 중 무엇을 더 타나요?", 
                ["더위를 못 참는다 (찬물 좋아함)", "추위를 못 참는다 (따뜻한 곳 좋아함)", "땀을 흘리면 개운하다", "땀 흘리면 기운이 빠진다"])
            
            q4 = st.radio("4. 평소 성격은 어떤가요?", 
                ["급하고 직선적이다", "꼼꼼하고 내성적이다", "느긋하고 참을성이 많다", "독창적이고 카리스마 있다"])

            if st.form_submit_button("진단 결과 보기"):
                with st.spinner("AI가 체질을 분석 중입니다..."):
                    # 답변 모음
                    answers = {"body": q1, "digestion": q2, "temp": q3, "sweat": q3, "mind": q4}
                    
                    # AI에게 판단 요청
                    result = analyze_constitution(answers)
                    
                    # 결과 저장 및 이동
                    db.register_user(st.session_state.user_id, result)
                    st.session_state.constitution = result
                    
                    st.success(f"분석 완료! 회원님은 **'{result}'** 성향이 강합니다.")
                    
                    # 2초 대기 후 이동 (여기서 에러가 났었습니다)
                    time.sleep(2) 
                    st.rerun()
        st.stop()

# 3. 진료실
st.subheader(f"🩺 {st.session_state.constitution} 맞춤 진료실")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어디가 불편하신가요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("증상을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("증상 분석 및 약초 검색 중..."):
            herbs = simple_search(prompt)
            diagnosis = generate_prescription(prompt, st.session_state.constitution, herbs)
            
            st.markdown(diagnosis)
            st.session_state.messages.append({"role": "assistant", "content": diagnosis})
            
            # DB 저장
            db.save_diagnosis(st.session_state.user_id, prompt, "AI 처방", diagnosis[:100])
