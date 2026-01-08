import streamlit as st
import os
import time
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # db.py 임포트

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("herb-knowledge")
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# --- 핵심 로직 함수 ---

def simple_search(query_text):
    """Pinecone 검색 및 가중치 적용"""
    try:
        # 아유르베다/인도 키워드 감지 시 쿼리 보정
        search_query = f"증상 '{query_text}'에 대한 약국 약, 한방 약초, 아유르베다 통합 처방"
        if any(kw in query_text for kw in ["인도", "아유르베다", "Ayurveda"]):
            search_query = f"Ayurveda 인도 아유르베다 핵심 허브 처방: {query_text}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
        payload = {"model": GEMINI_EMBED_MODEL, "content": {"parts": [{"text": search_query}]}, "taskType": "RETRIEVAL_QUERY"}
        res = requests.post(url, json=payload).json()
        vector = res['embedding']['values']
        
        # 상위 20개를 검색하여 카테고리 쏠림 방지
        results = index.query(vector=vector, top_k=20, include_metadata=True)
        
        contexts, debug_list = [], []
        for match in results['matches']:
            m = match['metadata']
            contexts.append(f"[{m.get('category')}] {m.get('name')}: {m.get('efficacy')}")
            debug_list.append(f"[{match['score']:.2f}] {m.get('name')} ({m.get('category')})")
        return "\n\n".join(contexts), debug_list
    except:
        return "데이터 검색 중 오류가 발생했습니다.", []

def generate_prescription(symptom, constitution, herb_list):
    """AI 최종 처방 생성"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    system_prompt = f"""
    당신은 통합 의학 전문가입니다. 환자의 체질({constitution})과 증상({symptom})을 분석하세요.
    [가이드라인]
    1. 급성 통증 완화는 [Pharmacy](약국약) 우선 추천.
    2. 장기적 체질 개선은 [동의보감] 및 [Ayurveda] 활용.
    3. {constitution} 체질의 금기 사항이나 주의할 점을 반드시 포함할 것.
    4. 혈자리(지압점) 추천을 포함하여 전문성을 높일 것.
    검색 데이터: {herb_list}
    """
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        res = requests.post(url, json=payload).json()
        return res['candidates'][0]['content']['parts'][0]['text']
    except:
        return "처방 생성에 실패했습니다. 다시 시도해 주세요."

# --- 화면 레이아웃 ---

st.set_page_config(page_title="AI 통합 체질 약국", page_icon="🌿", layout="wide")

# 1. 로그인 세션 관리
if "user_id" not in st.session_state:
    st.title("🌿 통합 AI 체질 맞춤 약국")
    with st.form("login"):
        uid = st.text_input("아이디(이름)를 입력하세요")
        if st.form_submit_button("입장"):
            if uid.strip():
                st.session_state.user_id = uid
                st.rerun()
    st.stop()

# 2. 체질 확인 (정밀 진단)
if "constitution" not in st.session_state:
    saved_c = db.get_user_constitution(st.session_state.user_id)
    if saved_c:
        st.session_state.constitution = saved_c
    else:
        st.subheader(f"🔍 {st.session_state.user_id}님, 정밀 체질 진단을 시작합니다.")
        with st.form("precision_quiz"):
            c1, c2 = st.columns(2)
            with c1:
                q1 = st.radio("1. 외모/체격", ["상체가 발달하고 목덜미가 굵다", "살이 찌기 쉽고 체구가 크다", "가슴 부위가 발달하고 걸음이 빠르다", "하체가 발달하고 체구가 아담하다"])
                q2 = st.radio("2. 평소 성격", ["추진력이 강하고 창의적이다", "참을성이 많고 보수적이다", "판단이 빠르고 명랑하다", "꼼꼼하고 내성적이다"])
            with c2:
                q3 = st.radio("3. 소화 상태", ["보통이다", "소화력이 매우 좋다", "소화는 잘되나 열이 잘 오른다", "자주 체하고 소화력이 약하다"])
                q4 = st.radio("4. 땀의 특징", ["소변이 시원해야 건강하다", "땀을 많이 흘려야 개운하다", "대변이 잘 나와야 개운하다", "땀을 많이 흘리면 기운이 없다"])
            q5 = st.radio("5. 추위/더위", ["추위보다 더위를 못 참는다", "보통이다", "더위보다 추위를 많이 탄다"])

            if st.form_submit_button("진단 완료"):
                # 간단 점수 합산 로직
                score = {"태양인": 0, "태음인": 0, "소양인": 0, "소음인": 0}
                if "목덜미" in q1: score["태양인"] += 1
                if "체구가 크다" in q1: score["태음인"] += 1
                if "가슴" in q1: score["소양인"] += 1
                if "하체" in q1: score["소음인"] += 1
                
                if "추진력" in q2: score["태양인"] += 1
                if "참을성" in q2: score["태음인"] += 1
                if "판단" in q2: score["소양인"] += 1
                if "꼼꼼" in q2: score["소음인"] += 1
                
                if "소화력이 약하다" in q3: score["소음인"] += 1
                if "기운이 없다" in q4: score["소음인"] += 1
                
                result = max(score, key=score.get)
                st.session_state.constitution = result
                db.save_user_constitution(st.session_state.user_id, result)
                st.rerun()
        st.stop()

# 3. 진료실 대화 화면
st.header(f"👨‍⚕️ {st.session_state.user_id}님 ({st.session_state.constitution}) 진료실")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("증상을 상세히 알려주세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("최적의 데이터를 검색 중..."):
            context, debug = simple_search(prompt)
            with st.sidebar:
                st.write("🔍 후보군 분석 (Top 20)")
                for d in debug: st.caption(d)
                
            ans = generate_prescription(prompt, st.session_state.constitution, context)
            st.markdown(ans)
            db.save_diagnosis(st.session_state.user_id, prompt, "통합 진료", ans[:100])
            st.session_state.messages.append({"role": "assistant", "content": ans})
