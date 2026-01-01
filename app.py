import streamlit as st
import os
import requests
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # 📂 구글 시트 연결 모듈 (기존에 쓰시던 것)

# 1. 로컬 환경(.env) 로드
load_dotenv()

# 2. 안전한 키 가져오기 함수
def get_secret(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("🚨 API 키 오류: .env 파일이나 Secrets를 확인하세요.")
    st.stop()

# Pinecone 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "herb-knowledge"
index = pc.Index(index_name)

# 모델 설정
GEMINI_EMBED_MODEL = "models/text-embedding-004"
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"
INTERVIEW_TURNS = 3 # 문진 횟수 조절

# --- Helper Functions ---

def get_gemini_embedding(text):
    # 🌟 핵심: taskType을 'RETRIEVAL_QUERY'로 설정 (이건 검색 질문이야! 라고 명시)
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
    payload = {
        "model": GEMINI_EMBED_MODEL,
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY" 
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['embedding']['values']
    except Exception as e:
        return None

def generate_gemini_response(messages, system_instruction):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    formatted_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        formatted_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": formatted_contents
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {e}"

def retrieve_context(query, top_k=5):
    try:
        # 검색어 보강: 단순히 '설사'가 아니라 '설사 증상에 좋은 약초'로 변환
        enhanced_query = f"증상 '{query}'에 효능이 있는 약초 정보"
        
        embedding = get_gemini_embedding(enhanced_query)
        if not embedding: return ""
        
        search_results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = []
        for match in search_results['matches']:
            meta = match['metadata']
            # 이름과 효능 위주로 컨텍스트 구성
            text = f"- 약초명: {meta.get('name', 'Unknown')}\n  효능: {meta.get('efficacy', '정보 없음')}\n  주의사항: {meta.get('caution', '정보 없음')}"
            contexts.append(text)
            
        return "\n\n".join(contexts)
    except Exception as e:
        return ""

# --- System Prompts (.replace 방식 사용으로 에러 원천 차단) ---

PROMPT_QUERY_REFINEMENT_DUAL = """
당신은 '검색 쿼리 최적화 전문가'입니다.
Pinecone 검색을 위해 **내복약용**과 **외용약용** 두 가지 쿼리를 각각 생성하세요.

[규칙]
1. **Line 1 (내복약)**: 증상 상세 + 체질 고려 키워드
2. **Line 2 (외용약)**: 증상 완화 + 찜질/아로마 키워드

[대화 내용]
__HISTORY__

[결과]
설명 없이 오직 **두 줄의 문자열만** 출력하세요.
"""

PROMPT_INTERVIEW = """
당신은 'AI 한의사'입니다. 
환자의 과거 기록(__HISTORY_CONTEXT__)을 참고하여 문진을 진행하세요.

[지침]
1. 과거에 방문한 적이 있다면, "지난번 __OLD_SYMPTOM__ 증상은 어떠신가요?"라고 안부를 먼저 물으세요.
2. 환자의 현재 불편한 증상을 구체적으로 파악하기 위해 질문하세요. (3회 이내)
3. 불필요한 인사는 생략하고 핵심만 질문하세요.
"""

PROMPT_PRESCRIPTION_EXPERT = """
당신은 명의(名醫) 'AI 한의사'입니다.
다음 검색된 약초 정보들을 바탕으로 환자에게 처방을 내리세요.

[검색된 약초 정보]
__CONTEXT_INTERNAL__
__CONTEXT_EXTERNAL__

[환자 증상]
__CHIEF_COMPLAINT__

[지침]
1. 검색된 약초 중에서 환자의 증상에 가장 적합한 것을 골라 **내복약**과 **외용법**을 추천하세요.
2. 약초의 이름과 효능을 구체적으로 언급하며 설명하세요.
3. 답변은 한국어로, 따뜻하고 전문적인 어조로 작성하세요.
"""

# --- Main App ---

st.set_page_config(page_title="심층 약초 상담소", page_icon="🌿", layout="centered")
st.markdown("<style>.stApp { background-color: #f6f7f2; color: #2e3b28; }</style>", unsafe_allow_html=True)
st.title("🌿 심층 약초 상담소")

# 로그인
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

if not st.session_state.patient_id:
    with st.form("login_form"):
        p_id = st.text_input("성함/전화번호 입력", placeholder="예: 홍길동1234")
        if st.form_submit_button("상담 시작"):
            if p_id:
                st.session_state.patient_id = p_id
                st.rerun()
    st.stop()

# 대화 기록 초기화
p_id = st.session_state.patient_id
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    history = db.get_patient_history(p_id)
    if history:
        last = history[-1]
        st.session_state.history_context = f"최근방문: {last['날짜']}, 증상: {last['증상']}"
        st.session_state.old_symptom = last['증상']
        greeting = f"반갑습니다 {p_id}님. 지난번 {last['증상']} 증상은 좀 어떠신가요? 오늘은 어디가 불편하신가요?"
    else:
        st.session_state.history_context = "신규 환자"
        st.session_state.old_symptom = "없음"
        greeting = f"반갑습니다 {p_id}님. 오늘 어디가 불편해서 오셨나요?"
    
    st.session_state.messages.append({"role": "assistant", "content": greeting})

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "diagnosis_complete" not in st.session_state:
    st.session_state.diagnosis_complete = False

# 채팅 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("답변을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 문진 단계
    if st.session_state.turn_count < INTERVIEW_TURNS:
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                final_prompt = PROMPT_INTERVIEW.replace("__HISTORY_CONTEXT__", st.session_state.history_context)
                final_prompt = final_prompt.replace("__OLD_SYMPTOM__", st.session_state.old_symptom)
                
                response = generate_gemini_response(st.session_state.messages, final_prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.turn_count += 1
        
        if st.session_state.turn_count >= INTERVIEW_TURNS:
             st.info("💡 진료를 위한 충분한 정보가 모였습니다. 다음 단계로 넘어갑니다.")

    # 진단 단계
    else:
        if not st.session_state.diagnosis_complete:
            with st.chat_message("assistant"):
                status = st.status("🔍 약초 데이터베이스 검색 중...", expanded=True)
                
                # 검색어 생성
                transcript = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                refine_prompt = PROMPT_QUERY_REFINEMENT_DUAL.replace("__HISTORY__", transcript)
                queries = generate_gemini_response([{"role": "user", "content": refine_prompt}], "")
                
                try:
                    q_lines = queries.strip().split('\n')
                    q_int = q_lines[0]
                    q_ext = q_lines[1] if len(q_lines) > 1 else q_int
                except:
                    q_int = "증상 완화 약초"
                    q_ext = "증상 완화 약초"
                
                status.write(f"검색 키워드: {q_int} / {q_ext}")

                # 검색 실행 (여기가 중요!)
                ctx_int = retrieve_context(q_int)
                ctx_ext = retrieve_context(q_ext)
                
                status.write("처방전 작성 중...")
                
                # 처방 생성
                symptom = st.session_state.messages[1]['content'] if len(st.session_state.messages) > 1 else "알 수 없음"
                final_prompt = PROMPT_PRESCRIPTION_EXPERT.replace("__CONTEXT_INTERNAL__", ctx_int)
                final_prompt = final_prompt.replace("__CONTEXT_EXTERNAL__", ctx_ext)
                final_prompt = final_prompt.replace("__CHIEF_COMPLAINT__", symptom)
                
                diagnosis = generate_gemini_response(st.session_state.messages, final_prompt)
                
                status.empty()
                st.markdown(diagnosis)
                
                # 저장
                db.save_diagnosis(p_id, symptom, "AI 진단", diagnosis[:200])
                
            st.session_state.messages.append({"role": "assistant", "content": diagnosis})
            st.session_state.diagnosis_complete = True
            
            if st.button("새로운 상담"):
                st.session_state.clear()
                st.rerun()
