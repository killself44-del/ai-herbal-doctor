import streamlit as st
import os
import requests
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import db  # 📂 구글 시트 모듈 (사용자님 기존 파일 유지)

# 1. 환경 설정
load_dotenv()

# 2. 안전한 키 가져오기
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

# Pinecone & Model 설정
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "herb-knowledge"
index = pc.Index(index_name)

GEMINI_EMBED_MODEL = "models/text-embedding-004"
GEMINI_GEN_MODEL = "gemini-2.0-flash-exp"

# --- 핵심 함수: 검색 (AI 생각 뺌) ---
def get_gemini_embedding(text):
    # 'RETRIEVAL_QUERY'로 설정하여 검색 최적화
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:embedContent?key={GOOGLE_API_KEY}"
    payload = {
        "model": GEMINI_EMBED_MODEL,
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY"
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()['embedding']['values']
    except:
        return None

def retrieve_context(user_symptom, top_k=5):
    try:
        # ⚡️ [수정] AI에게 검색어를 다시 만들라고 시키지 않고, 
        # 사용자가 말한 증상을 그대로 넣되, '문맥'만 살짝 추가합니다.
        query_text = f"증상 '{user_symptom}'을 치료하는 약초의 효능과 사용법"
        
        embedding = get_gemini_embedding(query_text)
        if not embedding: return ""
        
        # 검색 실행
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 결과 정리
        contexts = []
        for match in results['matches']:
            meta = match['metadata']
            # 이름과 효능을 가져옴
            text = f"약초명: {meta.get('name')}\n효능: {meta.get('efficacy')}\n주의사항: {meta.get('caution')}"
            contexts.append(text)
            
        return "\n\n".join(contexts)
    except Exception as e:
        return f"검색 오류: {e}"

def generate_diagnosis(messages, retrieved_info):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    # 시스템 프롬프트: 검색된 정보만 가지고 답하게 강제함
    system_prompt = f"""
    당신은 약초 처방 전문가입니다. 
    아래 [검색된 약초 정보]를 바탕으로 환자의 증상에 맞는 약을 추천하세요.
    
    [검색된 약초 정보]
    {retrieved_info}
    
    [주의사항]
    1. 반드시 위 **검색된 정보에 있는 약초** 중에서만 추천하세요.
    2. '호유'나 '파슬리'가 검색 결과에 없다면 절대 언급하지 마세요.
    3. 환자의 증상과 가장 잘 맞는 약초 1~2가지를 골라 이유를 설명하세요.
    """
    
    formatted_contents = [{"role": "user", "parts": [{"text": f"환자 증상: {messages[-1]['content']}"}]}]
    
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": formatted_contents
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"생성 오류: {e}"

# --- 메인 화면 ---
st.set_page_config(page_title="바른 약초 찾기", page_icon="🌿")
st.title("🌿 증상별 약초 처방 (직접 검색 모드)")

# 간단 로그인
if "patient_id" not in st.session_state:
    st.text_input("사용자 이름", key="input_id")
    if st.session_state.input_id:
        st.session_state.patient_id = st.session_state.input_id
        st.rerun()
    st.stop()

# 대화창
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어디가 불편하신가요? 증상을 말씀해주시면 딱 맞는 약초를 찾아드립니다."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("증상을 입력하세요 (예: 배가 아프고 설사가 나요)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        status = st.status("🔍 데이터베이스 뒤지는 중...", expanded=True)
        
        # 1. 검색 (DB에서 바로 가져오기)
        retrieved_herbs = retrieve_context(prompt)
        status.write("✅ 약초 데이터 확보 완료!")
        
        # 디버깅용: 실제로 뭘 가져왔는지 눈으로 확인 (중요!)
        with st.expander("🤖 AI가 찾아낸 후보 약초들 (클릭해서 확인)"):
            st.text(retrieved_herbs)
            
        # 2. 진단 생성
        status.write("📝 처방전 작성 중...")
        diagnosis = generate_diagnosis(st.session_state.messages, retrieved_herbs)
        status.update(label="진단 완료", state="complete", expanded=False)
        
        st.markdown(diagnosis)
        
        # 저장 (선택 사항)
        if hasattr(db, 'save_diagnosis'):
            db.save_diagnosis(st.session_state.patient_id, prompt, "약초 처방", diagnosis[:200])

    st.session_state.messages.append({"role": "assistant", "content": diagnosis})
