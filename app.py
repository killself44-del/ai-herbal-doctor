import streamlit as st
import os
import time
import requests
import hashlib
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # 우리가 수정한 db.py 모듈

# 1. 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Pinecone 및 모델 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("herb-knowledge")
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"  # 혹은 "gemini-1.5-pro"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# --- [함수 정의] ---

def simple_search(query_text):
    """Pinecone에서 관련 약재/약품 검색"""
    try:
        # 검색 품질 향상을 위한 쿼리 보정
        search_query = f"증상 '{query_text}'에 효과적인 약국 약, 한방 약초, 아유르베다 처방"
        if "인도" in query_text or "아유르베다" in query_text:
            search_query = f"아유르베다(Ayurveda) 인도 허브 처방: {query_text}"

        # 임베딩 생성
        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {
            "model": GEMINI_EMBED_MODEL,
            "content": {"parts": [{"text": search_query}]},
            "taskType": "RETRIEVAL_QUERY"
        }
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        # 상위 20개를 가져와서 약국/아유르베다가 밀리지 않게 함
        results = index.query(vector=vector, top_k=20, include_metadata=True)
        
        contexts = []
        debug_list = []
        for match in results['matches']:
            m = match['metadata']
            category = m.get('category', '미분류')
            name = m.get('name', '이름 없음')
            efficacy = m.get('efficacy', '정보 없음')
            
            contexts.append(f"[{category}] {name}: {efficacy}")
            debug_list.append(f"[{match['score']:.2f}] {name} ({category})")
            
        return "\n\n".join(contexts), debug_list
    except Exception as e:
        return f"검색 중 오류 발생: {e}", []

def generate_prescription(symptom, constitution, herb_list):
    """Gemini를 이용한 최종 처방전 생성"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    system_prompt = f"""
    당신은 통합 의학 전문가입니다. 환자의 체질({constitution})과 증상({symptom})을 분석하여 처방을 내리세요.
    
    [처방 가이드라인]
    1. 급성 통증이나 빠른 효과가 필요할 땐 검색된 [Pharmacy](약국약)를 우선 추천하세요.
    2. 체질 개선과 장기적 치료에는 [동의보감] 또는 [Ayurveda] 약재를 추천하세요.
    3. 소음인 체질 특성(몸이 차고 소화력이 약함)을 고려하여 성질이 너무 차가운 약재는 주의사항을 명시하세요.
    4. 가능한 경우 지압점(혈자리)이나 생활 습관 조언도 포함하세요.

    [검색된 데이터 리스트]
    {herb_list}
    
    형식은 1.급성처방, 2.체질맞춤대안, 3.주의사항 및 생활조언 순서로 작성하세요.
    """
    
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text']
    except:
        return "죄송합니다. 처방전을 생성하는 중에 문제가 발생했습니다."

# --- [UI 시작] ---

st.set_page_config(page_title="통합 AI 체질 약국", page_icon="🌿", layout="wide")

# 사이드바: 로고 및 정보
with st.sidebar:
    st.title("🌿 AI 통합 약국")
    st.info("동의보감, 아유르베다, 현대 약국 약을 통합하여 최적의 처방을 제공합니다.")
    st.divider()

# 1. 로그인 로직
if "user_id" not in st.session_state:
    st.subheader("🔑 로그인")
    with st.form("login_form"):
        user_id = st.text_input("아이디(성함)를 입력하세요")
        if st.form_submit_button("입장하기"):
            if user_id.strip():
                st.session_state.user_id = user_id
                st.rerun()
            else:
                st.warning("아이디를 입력해주세요.")
    st.stop()

# 2. 체질 진단 로직 (DB 연동)
if "constitution" not in st.session_state:
    with st.spinner("사용자 체질 정보를 불러오는 중..."):
        saved_c = db.get_user_constitution(st.session_state.user_id)
        
    if saved_c:
        st.session_state.constitution = saved_c
    else:
        st.subheader(f"🔍 {st.session_state.user_id}님, 체질 진단이 필요합니다.")
        
        with st.form("quiz_form"):
            q1 = st.radio("1. 평소 소화는 잘 되시나요?", ["자주 체하고 소화가 느리다", "보통이다", "소화력이 매우 좋고 금방 배고프다"])
            q2 = st.radio("2. 평소 몸의 온도는 어떤가요?", ["추위를 많이 타고 손발이 차다", "보통이다", "열이 많고 땀이 많다"])
            q3 = st.radio("3. 체형은 어떤 편이신가요?", ["상체에 비해 하체가 발달하고 아담하다", "골격이 굵고 체구가 큰 편이다", "상체가 발달하고 걸음걸이가 빠르다"])
            
            if st.form_submit_button("진단 완료"):
                # 간단 진단 알고리즘
                if "추위" in q2 or "자주 체" in q1:
                    result = "소음인"
                elif "열이 많" in q2 or "상체" in q3:
                    result = "소양인"
                else:
                    result = "태음인"
                
                st.session_state.constitution = result
                db.save_user_constitution(st.session_state.user_id, result)
                st.success(f"진단 결과 {result} 체질로 확인되었습니다!")
                time.sleep(1.5)
                st.rerun()
        st.stop()

# 3. 메인 진료실 화면
st.header(f"👨‍⚕️ {st.session_state.user_id}님 ({st.session_state.constitution}) 진료실")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 내역 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("어디가 불편하신가요? (예: 머리가 아프고 속이 울렁거려요)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("데이터베이스 분석 및 처방 구성 중..."):
            # 데이터 검색
            context_text, debug_info = simple_search(prompt)
            
            # 사이드바에 검색 결과 표시 (디버깅용)
            with st.sidebar:
                st.write(f"🔍 '{prompt}' 검색 결과 (Top 20)")
                for d in debug_info:
                    st.caption(d)
            
            # 처방전 생성
            response = generate_prescription(prompt, st.session_state.constitution, context_text)
            st.markdown(response)
            
            # DB에 진료 기록 저장
            db.save_diagnosis(st.session_state.user_id, prompt, "통합 진료", response[:100] + "...")
            
    st.session_state.messages.append({"role": "assistant", "content": response})
