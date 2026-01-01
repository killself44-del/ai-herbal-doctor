import streamlit as st
import os
import requests
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # 📂 구글 시트 연결 모듈

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
    st.error("🚨 API 키를 찾을 수 없습니다! Streamlit Cloud의 [Settings] -> [Secrets]에 키가 저장되었는지 확인해주세요.")
    st.stop()

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "herb-knowledge"
index = pc.Index(index_name)

# Constants
GEMINI_EMBED_MODEL = "models/text-embedding-004"
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"
INTERVIEW_TURNS = 4 # 심층 문진 횟수

# --- Helper Functions (기존 고급 로직 유지) ---

def get_gemini_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
    payload = {"model": GEMINI_EMBED_MODEL, "content": {"parts": [{"text": text}]}}
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

def retrieve_context(query, top_k=3):
    try:
        embedding = get_gemini_embedding(query)
        if not embedding: return ""
        
        search_results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = []
        for match in search_results['matches']:
            meta = match['metadata']
            text = f"- 이름: {meta.get('name', 'Unknown')}\n  효능: {meta.get('efficacy', '정보 없음')}\n  사용법: {meta.get('usage', '정보 없음')}\n  주의사항: {meta.get('caution', '정보 없음')}"
            contexts.append(text)
            
        return "\n\n".join(contexts)
    except Exception as e:
        return ""

# --- System Prompts (업그레이드: 교통정리 & DB 반영) ---

PROMPT_QUERY_REFINEMENT_DUAL = """
당신은 '검색 쿼리 최적화 전문가'입니다.
Pinecone 검색을 위해 **내복약용**과 **외용약용** 두 가지 쿼리를 각각 생성하세요.

[규칙]
1. **Line 1 (내복약)**: 증상 상세(80%) + 체질 보완(20%)
   - 예: "뒷목 뻣뻣함 긴장성 두통 갈근 (소음인 몸이 참)"
2. **Line 2 (외용약)**: 증상 완화 + 찜질/아로마 키워드
   - 예: "두통 쿨링 국화 박하 찜질 (진정 효과)"

[대화 내용]
{history}

[결과]
설명 없이 오직 **두 줄의 문자열만** 출력하세요.
"""

# ⭐ [핵심 수정] 과거 기록 반영 + 교통정리(Traffic Control) 기능 추가
# (수정됨) 문진 프롬프트: 체질 기억 & 완치 여부 확인 기능 강화
# (수정됨) 문진 프롬프트: 에러 수정본 (old_symptom 변수 제거)
PROMPT_INTERVIEW = """
당신은 기억력이 좋고 융통성 있는 'AI 한의사'입니다.
현재 단계는 **[심층 문진(Deep Interview) 단계]**입니다.

[환자의 과거 진료 기록]
{history_context}

[지침 1: 체질 정보 재사용 (중복 질문 금지)]
- 과거 기록에 환자의 **체질(예: 소음인, 몸이 참, 열이 많음 등)**에 대한 정보가 있다면, 이번 문진에서는 소화/대변/한열(추위탐) 등의 **체질 확인 질문을 생략**하세요.
- 대신 "환자분은 평소 몸이 차신 편이니..."와 같이 **알고 있다는 것을 언급**하며 공감대를 형성하세요.

[지침 2: 신규/구 증상 교통정리 (Traffic Control)]
환자가 오늘 **새로운 증상(New)**을 이야기했을 때:
1. **완치 여부 확인 (최우선)**: AI가 임의로 판단하지 말고, 반드시 **"그런데 지난번 불편하셨던 증상은 이제 깨끗이 나으셨습니까?"**라고 먼저 물어보세요. (기록된 과거 증상명을 언급하며 물어보세요)
2. **Case A (완치됨)**: 환자가 "나았다"고 하면, 과거 증상은 잊고 **오직 새로운 증상**에만 집중하여 진단하세요. (과거 처방에 구애받지 마세요)
3. **Case B (안 나음)**: 환자가 "아직 안 나았다"고 하면, 그때 비로소 **"두 가지 다 치료해야겠군요. 하지만 약효 집중을 위해 당장 더 괴로운 것 하나만 먼저 꼽아주시겠어요?"**라고 우선순위를 정하세요.

[문진 진행 순서]
1. **Turn 1**: 주증상 파악 및 **과거 증상 완치 여부 확인** (필수)
2. **Turn 2**: 통증의 상세 양상 및 악화 요인
3. **Turn 3**: (과거 기록에 체질 정보가 **없을 때만**) 전신 상태/체질 파악
   - *주의*: 체질 정보가 이미 있다면 Turn 3는 생략하고 바로 진단 단계로 넘어가겠다고 말하세요.
"""

# (수정됨) 처방 프롬프트: 체질 기록 저장 강화
PROMPT_PRESCRIPTION_EXPERT = """
당신은 명의(名醫) 'AI 한의사'입니다.
현재 단계는 진단 및 처방 단계입니다.

[입력 데이터]
1. 내복약 후보: {context_internal}
2. 외용약 후보: {context_external}
3. 환자 주증상: {chief_complaint}

[지침]
1. **체질 명시**: 진단 내용에 반드시 환자의 **추정 체질(예: 소음인 경향, 몸이 찬 체질 등)**을 텍스트로 남기세요. (다음 진료 때 기억하기 위함입니다)
2. **증상 분리**: 만약 환자가 "과거 증상은 다 나았다"고 했다면, 과거 증상용 약재는 빼고 **오직 오늘 증상**에 맞는 약재만 처방하세요.
3. 답변 포맷 (4단계):
   **1. 🩺 정밀 진단 (체질 분석 포함)**
      - "환자분의 기록과 증상을 종합할 때 [체질]으로 판단됩니다..."
   **2. 🍵 내복요법 (치료 중심)**
   **3. 🩹 외용요법 (안전 제일)**
   **4. 🧘 생활요법**

4. **주의**: 오직 한국어(Korean)만 사용하세요.
"""

# --- Main App ---

st.set_page_config(page_title="심층 약초 상담소", page_icon="🌿", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f6f7f2; color: #2e3b28; }
    h1 { color: #4a5d23; font-family: 'Malgun Gothic', sans-serif; text-align: center; margin-bottom: 2rem; }
    .stChatMessage { border-radius: 12px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stButton>button { background-color: #6b8c42 !important; color: white !important; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 심층 약초 상담소")

# --- [추가 기능 1] 로그인 시스템 ---
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

if not st.session_state.patient_id:
    with st.form("login_form"):
        st.subheader("📋 진료 접수")
        p_id = st.text_input("성함이나 전화번호 뒷자리를 입력하세요", placeholder="예: 홍길동1234")
        if st.form_submit_button("상담 시작"):
            if p_id:
                st.session_state.patient_id = p_id
                st.rerun()
    st.warning("⚠️ 본 정보는 참고용이며, 의학적 진단을 대신할 수 없습니다.")
    st.stop()

# 로그인 성공 후 로직
p_id = st.session_state.patient_id
st.sidebar.success(f"환자: {p_id}님 접속 중")

# 초기화 및 DB 기록 불러오기
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # DB 조회
    history = db.get_patient_history(p_id)
    
    if history:
        last = history[-1]
        st.session_state.history_context = f"- 최근방문: {last['날짜']}\n- 당시증상: {last['증상']}\n- 당시처방: {last['처방약재']}"
        greeting = f"반갑습니다 {p_id}님. 지난번({last['날짜']})엔 **'{last['증상']}'** 문제로 처방을 받으셨네요. 그간 차도는 좀 있으셨습니까? 오늘 불편하신 곳은 어디인지요?"
    else:
        st.session_state.history_context = "과거 진료 기록 없음 (신규 환자)"
        greeting = f"반갑습니다 {p_id}님, AI 한의사입니다.\n\n오늘 어떤 불편함 때문에 찾아오셨는지요? 증상을 자세히 말씀해 주시면 꼼꼼하게 살펴드리겠습니다."
    
    st.session_state.messages.append({"role": "assistant", "content": greeting})

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "diagnosis_complete" not in st.session_state:
    st.session_state.diagnosis_complete = False

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("증상을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # [Branch A] 심층 문진 모드 (Turn 1~4)
    if st.session_state.turn_count < INTERVIEW_TURNS:
        with st.chat_message("assistant"):
            with st.spinner("증상을 살피는 중입니다..."):
                # 프롬프트에 DB 기록(history_context) 주입
                final_interview_prompt = PROMPT_INTERVIEW.format(history_context=st.session_state.history_context)
                
                response_text = generate_gemini_response(
                    st.session_state.messages, 
                    final_interview_prompt
                )
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.turn_count += 1
        
        if st.session_state.turn_count >= INTERVIEW_TURNS:
             st.info("💡 충분한 정보가 모였습니다. 다음 단계에서 정밀 진단을 시작합니다.")

    # [Branch B] 정밀 진단 모드 (Turn >= 4)
    else:
        if not st.session_state.diagnosis_complete:
            with st.chat_message("assistant"):
                status_text = st.status("🔍 정밀 분석 중: 내복약과 외용약을 분리하여 검색합니다...", expanded=True)
                
                # 1. 쿼리 최적화 (Dual Query)
                transcript = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                refine_prompt = PROMPT_QUERY_REFINEMENT_DUAL.format(history=transcript)
                raw_queries = generate_gemini_response([{"role": "user", "content": refine_prompt}], "")
                
                try:
                    lines = [line.strip() for line in raw_queries.strip().split('\n') if line.strip()]
                    query_internal = lines[0] if len(lines) > 0 else "건강 상담"
                    query_external = lines[1] if len(lines) > 1 else query_internal
                except:
                    query_internal = "건강 상담"; query_external = "건강 상담"

                status_text.write(f"🔍 검색어 생성:\n1. 내복: {query_internal}\n2. 외용: {query_external}")
                
                # 2. Retrieve (Dual RAG)
                context_internal = retrieve_context(query_internal, top_k=3)
                context_external = retrieve_context(query_external, top_k=3)
                status_text.write("💊 안전성 검증 및 처방 작성 중...")
                
                # 3. Generate Prescription
                if len(st.session_state.messages) > 1:
                    # 보통 두 번째 메시지가 주증상
                    original_symptom = st.session_state.messages[1]['content']
                else:
                    original_symptom = "알 수 없음"

                final_system_prompt = PROMPT_PRESCRIPTION_EXPERT.format(
                    context_internal=context_internal, 
                    context_external=context_external,
                    chief_complaint=original_symptom
                )
                
                response_text = generate_gemini_response(
                    st.session_state.messages, 
                    final_system_prompt
                )
                
                status_text.empty()
                st.markdown(response_text)
                
                # --- [추가 기능 2] 자동 저장 ---
                # 진단 결과 앞부분만 요약해서 저장 (너무 길면 셀이 터지니까)
                if db.save_diagnosis(p_id, original_symptom, "AI 정밀 진단", response_text[:300]+"..."):
                    st.success("💾 진료 기록이 안전하게 저장되었습니다.")
                else:
                    st.error("⚠️ 저장 실패 (관리자에게 문의하세요)")
                
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.diagnosis_complete = True


