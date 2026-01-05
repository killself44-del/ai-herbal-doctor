import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import datetime

# 1. 인증 설정
def get_client():
    # secrets.toml 또는 로컬 json 파일 사용
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # 로컬용 (google_key.json 파일이 있다고 가정)
    try:
        creds = Credentials.from_service_account_file("google_key.json", scopes=scope)
    except:
        # Streamlit Cloud용 (Secrets에 google_key가 있는 경우)
        creds = Credentials.from_service_account_info(st.secrets["google_key"], scopes=scope)
        
    return gspread.authorize(creds)

# 2. 환자 체질 조회 (users 시트)
def get_user_constitution(user_id):
    try:
        client = get_client()
        sheet = client.open("약초_데이터베이스").worksheet("users") # 시트 이름 확인!
        
        # ID로 검색
        cell = sheet.find(user_id)
        if cell:
            # ID 옆 칸(체질)을 가져옴
            return sheet.cell(cell.row, cell.col + 1).value
        return None # 없으면 None 반환
    except Exception as e:
        return None

# 3. 신규 환자 등록 (users 시트)
def register_user(user_id, constitution):
    try:
        client = get_client()
        sheet = client.open("약초_데이터베이스").worksheet("users")
        
        # [ID, 체질, 가입일] 추가
        sheet.append_row([user_id, constitution, str(datetime.datetime.now())])
        return True
    except Exception as e:
        print(f"등록 에러: {e}")
        return False

# 4. 진료 기록 저장 (records 시트)
def save_diagnosis(user_id, symptom, diagnosis, prescription):
    try:
        client = get_client()
        sheet = client.open("약초_데이터베이스").worksheet("records")
        
        # [날짜, ID, 증상, 진단내용, 처방약재]
        sheet.append_row([
            str(datetime.datetime.now()), 
            user_id, 
            symptom, 
            diagnosis, 
            prescription
        ])
    except Exception as e:
        print(f"저장 에러: {e}")
