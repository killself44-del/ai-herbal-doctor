import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 1. 구글 시트 연결
def get_connection():
    # Secrets에서 키 정보 가져오기 (이름 주의: gcp_service_account)
    credentials_dict = st.secrets["gcp_service_account"]
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
    client = gspread.authorize(creds)
    # ⚠️ 중요: 구글 시트 파일 이름을 정확히 적으세요!
    sheet = client.open("AI한의사_진료기록부").sheet1
    return sheet

# 2. 진료 기록 저장 (쓰기)
def save_diagnosis(patient_id, symptoms, diagnosis, prescription):
    try:
        sheet = get_connection()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([date, patient_id, symptoms, diagnosis, prescription])
        return True
    except Exception as e:
        # 에러 나면 화면에 보여줌 (디버깅용)
        st.error(f"DB 저장 실패: {e}")
        return False

# 3. 과거 기록 조회 (읽기)
def get_patient_history(patient_id):
    try:
        sheet = get_connection()
        all_records = sheet.get_all_records()
        # 내 ID와 같은 기록만 찾아서 리스트로 만듦
        history = [row for row in all_records if str(row['환자ID']) == str(patient_id)]
        return history
    except:
        return []
