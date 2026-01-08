import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

# 구글 시트 인증 설정
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# 서비스 계정 키 파일 이름이 정확한지 확인하세요!
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)

# 구글 시트 열기 (본인의 스프레드시트 이름으로 수정)
spreadsheet_name = "AI_Pharmacy_DB" 
sheet = client.open(spreadsheet_name)

# 1. 사용자의 체질 정보 가져오기
def get_user_constitution(user_id):
    try:
        user_sheet = sheet.worksheet("users") # 탭 이름 확인
        cell = user_sheet.find(user_id)
        if cell:
            return user_sheet.cell(cell.row, 2).value # 2번째 열에 저장된 체질 반환
        return None
    except Exception as e:
        print(f"DB 읽기 에러: {e}")
        return None

# 2. 신규 사용자의 체질 정보 저장하기
def save_user_constitution(user_id, constitution):
    try:
        user_sheet = sheet.worksheet("users")
        # 이미 있는 아이디인지 확인 후 없으면 추가
        if not user_sheet.find(user_id):
            user_sheet.append_row([user_id, constitution])
            return True
        return False
    except Exception as e:
        print(f"DB 저장 에러: {e}")
        return False

# 3. 진료 기록 저장하기 (검색 기록)
def save_diagnosis(user_id, symptom, category, prescription):
    try:
        record_sheet = sheet.worksheet("diagnoses") # 탭 이름 확인
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_sheet.append_row([now, user_id, symptom, category, prescription])
        return True
    except Exception as e:
        print(f"기록 저장 에러: {e}")
        return False
