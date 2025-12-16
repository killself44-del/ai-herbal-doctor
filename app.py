import streamlit as st
import os
import requests
import json
from pinecone import Pinecone
from dotenv import load_dotenv

# 1. 로컬 환경(.env) 로드
load_dotenv()

# 2. 안전한 키 가져오기 함수 (에러 방지용)
def get_secret(key_name):
    # 1순위: Streamlit Cloud Secrets 시도
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass  # Secrets가 없어도 에러 내지 말고 넘어가!
    
    # 2순위: 로컬 환경변수 시도
    return os.getenv(key_name)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

# 3. 키가 없으면 친절하게 알려주기
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
INTERVIEW_TURNS = 4 # Increased for deeper symptom analysis

# --- Helper Functions (REST API) ---
# --- Helper Functions (REST API) ---

def get_gemini_embedding(text):
    """Get embedding using Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
    payload = {
        "model": GEMINI_EMBED_MODEL,
        "content": {"parts": [{"text": text}]}
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['embedding']['values']
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

def generate_gemini_response(messages, system_instruction):
    """
    Generate response using Gemini REST API.
    Args:
        messages: List of {"role": str, "content": str}
        system_instruction: String
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    # Construct strictly interleaved content: User -> Model -> User ...
    # System instruction can be passed in 'system_instruction' field for gemini-1.5/2.0
    
    formatted_contents = []
    
    # Add history
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
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error generating response: {e}\nRaw Response: {response.text if 'response' in locals() else 'None'}"

def retrieve_context(query, top_k=3):
    """Retrieve relevant documents from Pinecone."""
    try:
        embedding = get_gemini_embedding(query)
        if not embedding:
            return ""
        
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
        st.error(f"Search Error: {e}")
        return ""

# --- System Prompts ---

PROMPT_QUERY_REFINEMENT_DUAL = """
당신은 '검색 쿼리 최적화 전문가'입니다.
Pinecone 검색을 위해 **내복약용**과 **외용약용** 두 가지 쿼리를 각각 생성하세요.

[규칙]
1. **Line 1 (내복약)**: 증상 상세(80%) + 체질 보완(20%)
   - **중요**: 단순히 '두통'이 아니라, '뒷목 통증', '욱신거림', '소화불량 동반' 등 구체적인 양상을 포함하세요.
   - 예: "뒷목 뻣뻣함 긴장성 두통 갈근 (소음인 몸이 참)"
2. **Line 2 (외용약)**: 증상 완화 + 찜질/아로마 키워드
   - 예: "두통 쿨링 국화 박하 찜질 (진정 효과)"

[대화 내용]
{history}

[결과]
설명 없이 오직 **두 줄의 문자열만** 출력하세요.
"""

PROMPT_INTERVIEW = """
당신은 매우 꼼꼼한 'AI 한의사'입니다.
현재 단계는 **[심층 문진(Deep Interview) 단계]**입니다.
당신의 목표는 환자의 '주증상'을 현미경 보듯 자세히 파악한 뒤, 보조적으로 체질을 확인하는 것입니다.

[문진 순서 및 지침]
1. **Turn 1~2 (증상 심층 파악)**:
   - 주증상의 **위치** (예: 머리 앞/뒤/옆, 배 위/아래)
   - **통증의 양상** (예: 콕콕 쑤심, 묵직함, 당김, 시림)
   - **악화/완화 요인** (예: 찬 바람 맞으면 심해짐, 밤에 심해짐, 스트레스)
   - *주의*: 체질(추위, 소화)은 아직 묻지 마세요. 증상부터 파십시오.

2. **Turn 3~4 (전신 상태 및 체질)**:
   - 증상이 파악된 후, 소화/대변/추위탐/수면 등을 물어 체질을 추론하세요.

3. **공통 지침**:
   - 한 번에 1~2개의 질문만 하세요.
   - "아까 뒷목이 당긴다고 하셨는데..." 처럼 환자의 말을 인용하여 공감대(Rapport)를 형성하세요.
"""

PROMPT_PRESCRIPTION_EXPERT = """
당신은 명의(名醫) 'AI 한의사'입니다.
현재 단계는 진단 및 처방 단계입니다. **내복약**과 **외용약**을 구분하여 처방하세요.

[입력 데이터]
1. 내복약 후보(Internal Context): {context_internal}
2. 외용약 후보(External Context): {context_external}
3. 환자 주증상: {chief_complaint}

[지침]
1. **검증 단계(Self-Reflection)**:
   - **내복약**: 주증상에 효과가 있고 체질에 맞는 약초 선택.
   - **🔴 절대 금기**: 
     - **식품성 약재(쌀, 대추, 감초, 생강 등)** 만으로 구성된 처방을 내리지 마세요. 반드시 **치료 효능이 강한 약초(천궁, 당귀, 갈근 등)**를 메인으로 포함해야 합니다.
     - 주증상과 관련 없는 약초(단지 체질만 맞는 경우)는 제외하세요.
   - **외용약**: 주증상 완화에 도움이 되는 약초 선택. (자극성 약재 얼굴 도포 금지)

2. 답변 포맷 (4단계):

   **1. 🩺 정밀 진단과 병리 분석**
   - "환자분은 [체질]에 가까우나, 현재 호소하시는 통증은 [구체적 양상]에 해당합니다. 이는 [한의학적 원인 추론]으로 보입니다."
   
   **2. 🍵 내복요법 (치료 중심)**
   - 선택된 약초 2~3가지를 소개하세요.
   - **처방 근거**: "이 약초를 선택한 이유는 [증상]을 [어떻게] 치료하기 때문입니다"라고 전문가스럽게 설명하세요. (단순 나열 금지)
   - 탕약/차 레시피 제안.

   **3. 🩹 외용요법 (안전 제일)**
   - [External Context] 활용. 안전한 찜질/도포법 제안.

   **4. 🧘 생활요법**
   - 증상 완화를 위한 구체적 행동.

3. **안전 경고**: "이 정보는 참고용이며, 정확한 진단은 한의원에서 받으세요"라고 덧붙이세요.
4. **주의**: **오직 한국어(Korean)만 사용하세요.**
"""

# --- Main App ---

# Page Config
st.set_page_config(page_title="심층 약초 상담소", page_icon="🌿", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f6f7f2; color: #2e3b28; }
    h1 { color: #4a5d23; font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; text-align: center; margin-bottom: 2rem; }
    .stChatMessage { border-radius: 12px; padding: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    div[data-testid="stChatMessageContent"] { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; line-height: 1.6; }
    .stButton>button { background-color: #6b8c42 !important; color: white !important; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 심층 약초 상담소")
st.warning("⚠️ 본 정보는 참고용이며, 의학적 진단을 대신할 수 없습니다.")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial Greeting
    greeting = "반갑습니다. AI 한의사입니다. \n\n오늘 어떤 불편함 때문에 찾아오셨는지요? 증상을 자세히 말씀해 주시면, 제 꼼꼼하게 살펴드리겠습니다."
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

if "diagnosis_mode" not in st.session_state:
    st.session_state.diagnosis_mode = False

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("증상을 입력하세요..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Logic Branching
    
    # Branch A: Interview Mode (Turns 0 ~ INTERVIEW_TURNS-1)
    if st.session_state.turn_count < INTERVIEW_TURNS:
        with st.chat_message("assistant"):
            with st.spinner("증상을 살피는 중입니다..."):
                response_text = generate_gemini_response(
                    st.session_state.messages, 
                    PROMPT_INTERVIEW
                )
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.turn_count += 1
        
        # Check if next turn should be diagnosis
        if st.session_state.turn_count >= INTERVIEW_TURNS:
             st.info("💡 충분한 정보가 모였습니다. 다음 단계에서 정밀 진단을 시작합니다.")

    # Branch B: Diagnosis Mode (Turn >= INTERVIEW_TURNS)
    else:
        with st.chat_message("assistant"):
            status_text = st.empty()
            status_text.text("🔍 정밀 분석 중: 내복약과 외용약을 분리하여 검색합니다...")
            
            # 1. Refine Search Query (Dual)
            transcript = "\\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            
            refine_prompt = PROMPT_QUERY_REFINEMENT_DUAL.format(history=transcript)
            
            raw_queries = generate_gemini_response([{"role": "user", "content": refine_prompt}], "")
            
            # Parse Queries
            try:
                # Naive splitting by newline
                lines = [line.strip() for line in raw_queries.strip().split('\n') if line.strip()]
                query_internal = lines[0] if len(lines) > 0 else "건강 상담"
                query_external = lines[1] if len(lines) > 1 else query_internal
            except:
                query_internal = "건강 상담"
                query_external = "건강 상담"

            status_text.text(f"🔍 검색어 생성:\n1. 내복: {query_internal}\n2. 외용: {query_external}")
            
            # 2. Retrieve (Dual RAG)
            context_internal = retrieve_context(query_internal, top_k=3)
            context_external = retrieve_context(query_external, top_k=3)
            
            status_text.text("💊 안전성 검증 및 처방 작성 중...")
            
            # 3. Generate Prescription
            if len(st.session_state.messages) > 1:
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
                
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Reset capability?
        # st.button("새로운 상담 시작", on_click=lambda: st.session_state.clear())


