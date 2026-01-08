import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import datetime

# 1. 구글 시트 인증 설정 (Secrets 사용)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

try:
    # Streamlit Secrets에서 [gcp_service_account] 섹션을 읽어옵니다.
    creds_info = st.secrets["gcp_service_account"]
    # dict 형식을 바로 사용하여 파일 없이 인증합니다.
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client = gspread.authorize(creds)
    
    # 구글 시트 열기 (시트 이름 확인 필수)
    spreadsheet_name = "AI_Pharmacy_DB" 
    sheet = client.open(spreadsheet_name)
except Exception as e:
    st.error(f"⚠️ 구글 시트 인증 오류: {e}. Secrets 설정을 확인하세요.")

# 2. 사용자의 체질 정보 가져오기
def get_user_constitution(user_id):
    try:
        user_sheet = sheet.worksheet("users")
        cell = user_sheet.find(user_id)
        if cell:
            return user_sheet.cell(cell.row, 2).value
        return None
    except:
        return None

# 3. 신규 사용자의 체질 정보 저장하기
def save_user_constitution(user_id, constitution):
    try:
        user_sheet = sheet.worksheet("users")
        # 중복 저장 방지
        if not user_sheet.find(user_id):
            user_sheet.append_row([user_id, constitution])
            return True
        return False
    except:
        return False

# 4. 진료 기록 저장하기
def save_diagnosis(user_id, symptom, category, prescription):
    try:
        record_sheet = sheet.worksheet("diagnoses")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_sheet.append_row([now, user_id, symptom, category, prescription])
        return True
    except:
        return False
