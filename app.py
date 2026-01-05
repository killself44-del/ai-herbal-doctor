import streamlit as st
import os
import time  # 에러 방지용 필수 모듈
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
    
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text'].strip()
    except:
        return "알 수 없음"

# --- 기능 2: 단순 검색 (검색어 강화 + 디버깅 리스트 반환) ---
def simple_search(query_text):
    try:
        # 🌟 검색어 강화: 사용자의 말을 검색용 언어(효능 중심)로 확장
        enhanced_query = f"증상 '{query_text}'를 치료하고 통증을 완화하는 약초, 한약재, 일반의약품(약국약) 효능"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {
            "model": GEMINI_EMBED_MODEL, 
            "content": {"parts": [{"text": enhanced_query}]},
            "taskType": "RETRIEVAL_QUERY"
        }
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        # Top 10개 검색
        results = index.query(vector=vector, top_k=10, include_metadata=True)
        
        contexts = []
        debug_list = [] # 개발자 확인용
        
        for match in results['matches']:
            m = match['metadata']
            
            # 검색된 텍스트 구성
            text = f"""
            - 이름: {m.get('name')}
            - 분류: {m.get('category')}
            - 효능: {m.get('efficacy')}
            - 주의사항: {m.get('caution')}
            - 체질/도샤: {m.get('dosha', '')}
            """
            contexts.append(text)
            
            # 디버깅 정보 저장 (점수, 이름, 분류)
            debug_list.append(f"[{match['score']:.2f}] {m.get('name')} ({m.get('category')})")
            
        return "\n\n".join(contexts), debug_list
    except Exception as e:
        return "", [f"에러 발생: {e}"]

# --- 기능 3: 맞춤 처방 (엄격 모드 적용) ---
def generate_prescription(symptom, constitution, herb_list):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    system_prompt = f"""
    당신은 엄격하고 정직한 '통합 의학 전문가'입니다.
    검색된 약초 목록을 검토하여 환자의 **체질({constitution})**과 **증상({symptom})**에 **'직접적으로'** 효과가 있는 것만 처방하세요.
    
    [환자 정보]
    - 체질: {constitution}
    - 호소 증상: {symptom}
    
    [검색된 데이터베이스 후보군]
    {herb_list}
    
    [🚨 절대 준수 사항 (Strict Rules)]
    1. **연관성 검증:** 후보군에 있는 약이 '{symptom}' 증상에 명확한 효능이 없다면 **절대 추천하지 마세요.**
       - 예: 두통 환자에게 소화제나 파슬리, 연고 등을 추천하지 말 것.
    2. **솔직함:** 만약 검색된 목록에 적합한 약이 단 하나도 없다면, 억지로 지어내지 말고 **"죄송합니다. 현재 데이터베이스에 해당 증상을 치료할 적합한 약초/약품 정보가 부족합니다."**라고 말하세요.
    3. **우선순위:**
       - 1순위: 증상 완화에 탁월한 **약국약(Pharmacy)** 또는 **전문 한약재**.
       - 2순위: 체질에 맞는 **아유르베다 허브**.
       
    [답변 양식]
    안녕하세요 {constitution} 환자분. 
    
    **1. 💊 추천 처방 (약국약/한방)**
    - 이름: ...
    - 이유: ...
    
    **2. 🌿 체질 맞춤 대안 (자연요법)**
    - 이름: ...
    - 이유: ...
    
    **3. ⚠️ 주의사항**
    - ...
    """
    
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    
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
    saved_const = db.get_user_constitution(st.session_state.user_id)
    
    if saved_const:
        st.session_state.constitution = saved_const
        st.toast(f"환영합니다! {saved_const} 체질의 {st.session_state.user_id}님", icon="✅")
    else:
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
                    answers = {"body": q1, "digestion": q2, "temp": q3, "sweat": q3, "mind": q4}
                    result = analyze_constitution(answers)
                    db.register_user(st.session_state.user_id, result)
                    st.session_state.constitution = result
                    st.success(f"분석 완료! 회원님은 **'{result}'** 성향이 강합니다.")
                    time.sleep(2)
                    st.rerun()
        st.stop()

# 3. 진료실
st.subheader(f"🩺 {st.session_state.constitution} 맞춤 진료실")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어디가 불편하신가요? (예: 머리가 깨질 듯이 아파요)"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("증상을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("증상 분석 및 약초 검색 중..."):
            # 검색 실행 (결과 텍스트 + 디버깅 정보)
            herbs_text, debug_info = simple_search(prompt)
            
            # 🕵️‍♂️ 사이드바에 디버깅 정보 표시
            with st.sidebar:
                st.markdown("---")
                with st.expander(f"🔍 '{prompt}' 검색 결과 (개발자용)", expanded=True):
                    if debug_info:
                        for item in debug_info:
                            st.caption(item)
                    else:
                        st.error("검색 결과 없음")

            # 처방 생성
            diagnosis = generate_prescription(prompt, st.session_state.constitution, herbs_text)
            
            st.markdown(diagnosis)
            st.session_state.messages.append({"role": "assistant", "content": diagnosis})
            
            # DB 저장
            db.save_diagnosis(st.session_state.user_id, prompt, "AI 처방", diagnosis[:100])
