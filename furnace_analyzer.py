import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF
import tempfile
import os
from datetime import timedelta
import numpy as np
import re # 파일 이름 파싱을 위해 re 모듈 추가

# ---------------------------------------------------------
# 1. 앱 설정 및 폰트
# ---------------------------------------------------------
st.set_page_config(page_title="가열로 다중 분석 시스템", layout="wide")

# 폰트 설정
FONT_FILE = 'NanumGothic.ttf'
HAS_KOREAN_FONT = False
try:
    if os.path.exists(FONT_FILE):
        font_prop = fm.FontProperties(fname=FONT_FILE)
        plt.rcParams['font.family'] = font_prop.get_name()
        HAS_KOREAN_FONT = True
    else:
        # 폰트 파일이 없는 경우, 기본 폰트 설정 유지 (대부분의 시스템에서 산세리프 폰트로 대체됨)
        plt.rcParams['font.family'] = 'sans-serif'
except Exception:
    plt.rcParams['font.family'] = 'sans-serif'
    
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

# ---------------------------------------------------------
# 2. 로직: 헤더 찾기 & 데이터 로딩
# ---------------------------------------------------------
@st.cache_data
def smart_read_file(uploaded_file, header_row=0, nrows=None):
    try:
        # Streamlit 환경에서는 파일 객체를 다시 읽기 위해 seek(0) 필요
        uploaded_file.seek(0) 
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            # header=None으로 읽어온 후, 지정된 행을 컬럼으로 설정하여 유연성 확보
            df = pd.read_excel(uploaded_file, header=None, nrows=nrows + header_row + 1 if nrows else None)
        else:
            uploaded_file.seek(0)
            try:
                # 엑셀 파일이 아닌 경우 (CSV)
                df = pd.read_csv(uploaded_file, encoding='cp949', header=None, nrows=nrows + header_row + 1 if nrows else None)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8', header=None, nrows=nrows + header_row + 1 if nrows else None)
        
        # 지정된 행을 컬럼 헤더로 설정
        if header_row < len(df):
             # 헤더 행으로 컬럼 이름 설정하고 그 이전 행들은 제거
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
            # 컬럼 이름이 중복되거나 None인 경우 처리
            df.columns = [f"{col}_{i}" if col is None else str(col).strip() for i, col in enumerate(df.columns)]
        
        return df
    except Exception as e: 
        st.error(f"파일 읽기 오류: {e}")
        return None

# ---------------------------------------------------------
# 3. 핵심 로직: 사이클 감지 및 분석
# ---------------------------------------------------------
def analyze_cycle(daily_data, temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end, check_strict_start):
    """
    조건:
    1. 시작: temp_start 이하에서 승온이 시작되는 지점 (장입 후 승온)
    2. 홀딩: temp_holding_min ~ temp_holding_max 구간이 duration_holding_min 이상 지속
    3. 종료: 홀딩 이후 temp_end 이하로 떨어지는 시점
    4. 유효성: (선택 사항) 시작 2시간 후부터 종료 시점까지 temp_start 미만으로 떨어지지 않아야 함
    """
    # 1. 시작점 찾기 
    
    start_row = None
    
    if check_strict_start:
        # **장입 후 승온 로직:** temp_start 이하로 떨어진 후 다시 급격히 승온되는 지점을 시작점으로 간주
        daily_data['temp_diff'] = daily_data['온도'].diff().fillna(0)
        
        # 1. temp_start 이하로 온도가 떨어진 지점 (장입 완료)
        low_temp_indices = daily_data[daily_data['온도'] <= temp_start].index
        
        # 2. low_temp_indices 이후의 급격한 상승 지점 (승온 시작)
        for idx in low_temp_indices:
            # 다음 10분간의 평균 온도 변화율이 일정 수준 이상인지 확인 (승온 시작)
            window = daily_data.loc[idx:idx + 10]
            if len(window) < 5: continue
            
            # 5분 동안 5도 이상 상승하는 지점을 승온 시작으로 간주
            if (window['온도'].iloc[-1] - window['온도'].iloc[0]) >= 5: # [이전 오류 수정: window['온도가'] -> window['온도']]
                # 시작 온도는 소재 장입이 완료된 후 온도가 상승하기 시작하는 시점
                start_row = daily_data.loc[idx]
                break
        
        if start_row is None:
            return None, "장입 후 유효한 승온 시작 지점 없음"
            
    else:
        # 기존 로직: temp_start 이하의 첫 지점을 시작점으로 간주
        start_candidates = daily_data[daily_data['온도'] <= temp_start]
        if start_candidates.empty:
            return None, f"시작 온도({temp_start}도 이하) 없음"
        start_row = start_candidates.iloc[0]

    start_time = start_row['일시']

    # 2. 홀딩 구간 찾기 (이하 기존 로직과 동일)
    post_start_data = daily_data[daily_data['일시'] > start_time].copy()
    
    # 홀딩 조건 마킹
    post_start_data['is_holding'] = (post_start_data['온도'] >= temp_holding_min) & (post_start_data['온도'] <= temp_holding_max)
    
    # 연속된 홀딩 구간 그룹화
    post_start_data['group'] = (post_start_data['is_holding'] != post_start_data['is_holding'].shift()).cumsum()
    
    holding_end_time = None
    
    # 각 그룹별 지속시간 체크
    duration_min_td = timedelta(hours=duration_holding_min)
    for _, group in post_start_data[post_start_data['is_holding']].groupby('group'):
        # 연속된 홀딩 기간의 시작과 끝
        if not group.empty:
            duration = group['일시'].max() - group['일시'].min()
            if duration >= duration_min_td:
                holding_end_time = group['일시'].max()
                break # 첫 번째 유효 홀딩 구간을 찾으면 중단
            
    if holding_end_time is None:
        return None, f"유효 홀딩 구간({duration_min_td} 이상) 없음" # duration_min_td 문자열로 표시

    # 3. 종료점 찾기
    post_holding_data = daily_data[daily_data['일시'] > holding_end_time]
    end_candidates = post_holding_data[post_holding_data['온도'] <= temp_end]
    
    if end_candidates.empty:
        return None, f"종료 온도({temp_end}도 이하) 도달 안 함"
        
    end_row = end_candidates.iloc[0]
    end_time = end_row['일시']

    # 4. 사이클 시작 후 2시간 이후에 비정상적인 저온 발생 여부 확인 (check_strict_start가 True일 때만 실행)
    if check_strict_start:
        # 2시간 후의 시작 시점 정의
        check_start_time = start_time + timedelta(hours=2)
        
        # 체크 윈도우: 시작 2시간 후부터 종료 시간 직전까지의 데이터 추출
        cycle_window = daily_data[(daily_data['일시'] >= check_start_time) & (daily_data['일시'] < end_time)].copy()

        # 이 구간 내에서 시작 온도(temp_start)보다 엄격하게 낮은 온도가 있는지 확인
        abnormal_low_temp = cycle_window[cycle_window['온도'] < temp_start]
        
        if not abnormal_low_temp.empty:
            abnormal_time = abnormal_low_temp.iloc[0]['일시'].strftime('%Y-%m-%d %H:%M')
            return None, f"사이클 시작 2시간 후 비정상적인 저온 발생 (<{temp_start}℃) at {abnormal_time}"
    
    return {
        'start_row': start_row,
        'end_row': end_row,
        'holding_end': holding_end_time
    }, "성공"

# 파일 이름에서 가열로 ID를 추출하는 헬퍼 함수
def extract_furnace_id_from_filename(filename):
    """파일 이름에서 '가열로X호기' 또는 '가열로X' 패턴을 찾아 ID를 추출합니다."""
    # '가열로'로 시작하고 '호기'로 끝나는 패턴 또는 '가열로X' 패턴을 찾습니다.
    # 예: 가열로 1호기_data.csv -> 가열로1호기
    match = re.search(r'(가열로\s*\d+\s*호기|가열로\s*\d+)', filename, re.IGNORECASE)
    if match:
        # 찾은 문자열에서 공백을 제거하고 반환
        return match.group(0).strip().replace(' ', '')
    return None

def process_data(sensor_files, df_prod, col_p_start_time, col_p_weight, col_p_unit, 
                 s_header_row, col_s_time, col_s_temp, col_s_gas,
                 target_cost, temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end, check_strict_start, use_target_cost, time_tolerance_hours): # check_charging_end, time_tolerance_hours 인자 추가
    
    # --- 생산실적 데이터 전처리 ---
    try:
        # col_p_date 대신 col_p_start_time 사용, 컬럼명을 '시작일시'로 변경
        df_prod = df_prod.rename(columns={col_p_start_time: '시작일시', col_p_weight: '장입량', col_p_unit: '가열로'})
        df_prod['시작일시'] = pd.to_datetime(df_prod['시작일시'], errors='coerce') # 시간 정보 유지
        if df_prod['장입량'].dtype == object:
            df_prod['장입량'] = df_prod['장입량'].astype(str).str.replace(',', '')
        df_prod['장입량'] = pd.to_numeric(df_prod['장입량'], errors='coerce')
        df_prod = df_prod.dropna(subset=['시작일시', '장입량', '가열로']).sort_values('시작일시')
    except Exception as e: return None, None, f"생산실적 오류: {e}"

    # --- 센서 데이터 통합 및 전처리 (이전과 동일) ---
    df_list = []
    for f in sensor_files:
        f.seek(0) # 파일 포인터 초기화
        df = smart_read_file(f, s_header_row)
        
        if df is not None:
            
            # 1. 파일 이름에서 가열로 ID 추출
            unit_id = extract_furnace_id_from_filename(f.name)
            if not unit_id:
                st.warning(f"경고: 센서 파일 {f.name}에서 유효한 가열로 ID를 찾을 수 없습니다. (패턴: 가열로X호기 또는 가열로X). 이 파일은 분석에서 제외됩니다.")
                continue

            try:
                # 2. 컬럼 이름 정규화
                df.columns = [str(c).strip() for c in df.columns]

                # 3. 컬럼 매핑
                df = df.rename(columns={col_s_time: '일시', col_s_temp: '온도', col_s_gas: '가스지침'})

                # 4. 가열로 ID 컬럼 추가
                df['가열로'] = unit_id

                # 5. 타입 변환 및 정리
                df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
                df['온도'] = pd.to_numeric(df['온도'], errors='coerce')
                df['가스지침'] = pd.to_numeric(df['가스지침'], errors='coerce')
                
                # 시간 컬럼 기준으로 정렬하고 NaN 제거
                df = df.dropna(subset=['일시', '가열로']).sort_values('일시')
                
                # 중복 일시 제거 (가장 마지막 값 유지)
                df = df.drop_duplicates(subset=['일시', '가열로'], keep='last').reset_index(drop=True)
                
                df_list.append(df)
            except Exception as e:
                st.error(f"센서 데이터 매핑 오류 (파일: {f.name}): {e}")
                
    if not df_list: return None, None, "센서 데이터 없음"
    
    df_sensor = pd.concat(df_list, ignore_index=True)
    
    # --- 다중 가열로 분석 실행 ---
    unit_ids = df_sensor['가열로'].unique()
    
    if len(unit_ids) > 20:
        return None, None, f"분석 대상 가열로가 {len(unit_ids)}개 감지되었습니다. 최대 20개까지만 분석을 지원합니다."
    
    if len(unit_ids) == 0:
        return None, None, "유효한 가열로 ID가 센서 데이터에서 발견되지 않았습니다."

    results = []
    
    for unit_id in unit_ids:
        # 1. 가열로별 데이터 필터링
        df_sensor_unit = df_sensor[df_sensor['가열로'] == unit_id].copy()
        df_prod_unit = df_prod[df_prod['가열로'] == unit_id].copy()
        
        if df_prod_unit.empty: continue # 생산 실적이 없는 가열로는 분석 제외

        # 2. 생산 실적 (차지별) 반복
        # 생산 실적의 모든 행(차지)을 기준으로 센서 데이터에서 유효 사이클을 찾습니다.
        for index, prod_row in df_prod_unit.iterrows():
            
            prod_start_time = prod_row['시작일시']
            
            # 생산 실적 시작 시간 주변의 48시간 윈도우 (센서 데이터)
            # 센서 사이클 시작은 생산 실적 시작보다 앞서거나 뒤쳐질 수 있으므로, 매칭 허용 시간만큼 앞뒤로 윈도우 설정
            window_start = prod_start_time - timedelta(hours=time_tolerance_hours)
            window_end = prod_start_time + timedelta(hours=48) # 충분히 긴 탐색 시간 (홀딩 시간 고려)
            
            daily_window = df_sensor_unit[
                (df_sensor_unit['일시'] >= window_start) & 
                (df_sensor_unit['일시'] < window_end) 
            ].copy()
            
            if daily_window.empty: continue
            
            # --- 해당 윈도우 내에서 가장 가까운 유효 사이클 찾기 ---
            
            temp_data = daily_window.copy()
            
            # 사이클 분석 수행 (첫 번째 유효 사이클만 찾음)
            cycle_info, msg = analyze_cycle(temp_data, temp_start, temp_holding_min, temp_holding_max, duration_min_td, temp_end, check_strict_start) # check_charging_end와 check_abnormal_low를 check_strict_start 하나로 통합
            
            if not cycle_info:
                continue # 유효 사이클 없음
            
            start = cycle_info['start_row']
            end = cycle_info['end_row']
            start_time_of_cycle = start['일시']

            # 매칭 검증: 센서 사이클 시작 시간과 생산 실적 시작 시간의 차이 확인
            match_diff = abs(prod_start_time - start_time_of_cycle)
            
            if match_diff > timedelta(hours=time_tolerance_hours):
                # 매칭 실패 (허용 범위 초과)
                continue
            
            # 3. 원단위 및 결과 계산
            
            charge_kg = prod_row['장입량']
            
            if charge_kg <= 0: continue
            
            gas_used = end['가스지침'] - start['가스지침']
            if gas_used <= 0: continue
            
            unit = gas_used / (charge_kg / 1000) # Nm3 / ton
            
            # 목표 원단위 사용 여부에 따라 달성 여부 설정
            if use_target_cost:
                is_pass = unit <= target_cost
                achievement = 'Pass' if is_pass else 'Fail'
            else:
                achievement = 'N/A' # 목표 원단위를 사용하지 않을 경우
            
            results.append({
                '가열로': unit_id,
                '날짜': start_time_of_cycle.strftime('%Y-%m-%d'),
                '검침시작': start_time_of_cycle.strftime('%Y-%m-%d %H:%M'),
                '시작지침': start['가스지침'],
                '검침완료': end['일시'].strftime('%Y-%m-%d %H:%M'),
                '종료지침': end['가스지침'],
                '가스사용량(Nm3)': int(gas_used),
                '장입량(kg)': int(charge_kg),
                '원단위': round(unit, 2),
                '달성여부': achievement,
                '비고': f"홀딩종료: {cycle_info['holding_end'].strftime('%H:%M')}"
            })

    # 전체 센서 데이터 반환 (필터링되지 않은 원본)
    return pd.DataFrame(results), df_sensor, None

# ---------------------------------------------------------
# 4. PDF 생성
# ---------------------------------------------------------
class PDFReport(FPDF):
    def __init__(self, unit_name, *args, **kwargs): # unit_name 추가
        self.unit_name = unit_name
        super().__init__(*args, **kwargs)
        if HAS_KOREAN_FONT: self.add_font('Nanum', '', FONT_FILE, uni=True)

    def header(self):
        font = 'Nanum' if HAS_KOREAN_FONT else 'Arial'
        self.set_font(font, 'B' if not HAS_KOREAN_FONT else '', 14)
        # 가열로 이름 동적 사용
        self.cell(0, 10, f"3. 가열로 {self.unit_name} 검증 DATA (개선 후)", 0, 1, 'L')
        self.ln(5)

def generate_pdf(row_data, chart_path, target, unit_name, use_target_cost): # use_target_cost 인자 추가
    pdf = PDFReport(unit_name=unit_name) # unit_name 전달
    pdf.add_page()
    font = 'Nanum' if HAS_KOREAN_FONT else 'Arial'
    
    pdf.set_font(font, '', 12)
    # 가열로 이름 동적 사용
    pdf.cell(0, 10, f"3.5 가열로 {unit_name} - {row_data['날짜']} (23% 절감 검증)", 0, 1, 'L')
    pdf.ln(5)

    pdf.set_fill_color(240, 240, 240)
    pdf.set_font(font, '', 10)
    headers = ["검침 시작", "검침 완료", "③ 가스 사용량\n(②-①=③)", "Cycle 종료", "장입량"]
    widths = [38, 38, 38, 38, 38]
    
    x = pdf.get_x(); y = pdf.get_y()
    for i, h in enumerate(headers):
        pdf.set_xy(x + sum(widths[:i]), y)
        pdf.multi_cell(widths[i], 6, h, border=1, align='C', fill=True)
    
    pdf.set_xy(x, y + 12)
    
    s_txt = f"{row_data['검침시작']}\n({row_data['시작지침']:,.0f})"
    e_txt = f"{row_data['검침완료']}\n({row_data['종료지침']:,.0f})"
    
    # Cycle 종료는 검침완료와 동일하게 표시 (비고의 홀딩 종료와 구분)
    vals = [s_txt, e_txt, f"{row_data['가스사용량(Nm3)']:,} Nm3", str(row_data['검침완료']), f"{row_data['장입량(kg)']:,} kg"]
    
    for i, v in enumerate(vals):
        cx = x + sum(widths[:i])
        pdf.set_xy(cx, y + 12)
        pdf.multi_cell(widths[i], 6, v, border=1, align='C')
        
    pdf.ln(5)
    pdf.set_y(y + 12 + 15)
    
    pdf.set_font(font, '', 12)
    pdf.cell(0, 10, "▶ 열처리 Chart (온도/가스 트렌드)", 0, 1, 'L')
    pdf.image(chart_path, x=10, w=190)
    
    pdf.ln(5)
    pdf.set_font(font, '', 10)
    
    # 목표 원단위 사용 여부에 따라 문구 변경
    if use_target_cost:
        report_footer = f"* 실적 원단위: {row_data['원단위']} Nm3/ton (목표 {target} 이하 달성)"
    else:
        report_footer = f"* 실적 원단위: {row_data['원단위']} Nm3/ton"

    pdf.cell(0, 8, report_footer, 0, 1, 'R')
    
    return pdf

# ---------------------------------------------------------
# 4.5 차트 생성 함수 (미리보기 및 PDF용)
# ---------------------------------------------------------
def plot_cycle_chart(row, full_raw, temp_holding_min, temp_holding_max, fig_width=10, fig_height=5):
    """주어진 사이클 정보를 바탕으로 Matplotlib 차트를 생성하여 반환합니다."""
    s_ts = pd.to_datetime(row['검침시작'])
    e_ts = pd.to_datetime(row['검침완료'])
    unit_id = row['가열로']
    
    # 전체 데이터에서 해당 가열로의 데이터만 필터링
    unit_raw = full_raw[full_raw['가열로'] == unit_id].copy()
    
    # 앞뒤로 1시간 여유 두기
    chart_data = unit_raw[(unit_raw['일시'] >= s_ts - timedelta(hours=1)) & (unit_raw['일시'] <= e_ts + timedelta(hours=1))].copy()
    
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    
    # 온도 트렌드
    ax1.fill_between(chart_data['일시'], chart_data['온도'], color='red', alpha=0.3)
    ax1.plot(chart_data['일시'], chart_data['온도'], 'r-', label='온도')
    ax1.set_ylabel('온도 (°C)', color='r')
    
    # 홀딩 구간 표시선
    ax1.axhline(y=temp_holding_min, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=temp_holding_max, color='gray', linestyle=':', alpha=0.5)
    
    # 가스 지침 트렌드
    ax2 = ax1.twinx()
    ax2.plot(chart_data['일시'], chart_data['가스지침'], 'b-', label='가스지침')
    ax2.set_ylabel('가스지침 (Nm3)', color='b')
    
    # 시작/종료 포인트 마커
    start_temp = chart_data.loc[chart_data['일시']>=s_ts, '온도'].iloc[0] if not chart_data.loc[chart_data['일시']>=s_ts, '온도'].empty else np.nan
    end_temp = chart_data.loc[chart_data['일시']<=e_ts, '온도'].iloc[-1] if not chart_data.loc[chart_data['일시']<=e_ts, '온도'].empty else np.nan
    ax1.scatter([s_ts, e_ts], [start_temp, end_temp], color='green', s=100, zorder=5)
    
    plt.title(f"가열로 {unit_id} Cycle: {row['검침시작']} ~ {row['검침완료']}")
    fig.autofmt_xdate() # X축 날짜 겹침 방지
    
    return fig

# ---------------------------------------------------------
# 4.6 컬럼 선택을 위한 헬퍼 함수
# ---------------------------------------------------------
def get_default_index(columns, keywords):
    """컬럼 이름에 키워드가 포함되어 있는지 확인하여 가장 적절한 기본 인덱스를 반환합니다."""
    for keyword in keywords:
        for i, col in enumerate(columns):
            # 컬럼 이름을 소문자로 변환하여 키워드 포함 여부 확인
            if keyword in str(col).lower():
                return i
    # 키워드 일치 항목이 없으면 첫 번째 컬럼을 기본값으로 반환
    return 0 

# ---------------------------------------------------------
# 5. 메인 UI
# ---------------------------------------------------------
def main():
    
    with st.sidebar:
        st.header("1. 데이터 업로드")
        
        prod_file = st.file_uploader("생산 실적 (Excel) - 가열로 ID 컬럼 필수", type=['xlsx'])
        st.info("센서 데이터는 파일 이름에서 가열로 ID를 자동으로 인식합니다. (예: 가열로X호기 또는 가열로X)")
        sensor_files = st.file_uploader("가열로 데이터 (CSV/Excel) - 파일 이름에서 ID 인식", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)
        
        st.divider()
        st.header("2. 분석 기준 설정")
        
        # 목표 원단위 사용 여부 체크박스 추가
        use_target_cost = st.checkbox("목표 원단위 사용 (Pass/Fail 분석)", value=True)

        if use_target_cost:
            target_cost = st.number_input("목표 원단위 (Nm3/ton)", value=25.53, step=0.1, format="%.2f")
        else:
            target_cost = None
            st.warning("목표 원단위 분석을 사용하지 않습니다. '달성여부'는 N/A로 표시됩니다.")
        
        # --- 사이클 정의 간소화 ---
        st.divider()
        st.header("🔥 사이클 정의 (최소 조건)")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            temp_start = st.number_input("시작 온도 (Max)", value=600, step=10, help="이 온도 이하일 때 사이클 시작 후보로 간주")
            temp_holding_min = st.number_input("홀딩 온도 (Min)", value=1230, step=10)
            temp_end = st.number_input("종료 온도 (Max)", value=900, step=10)
        with col_t2:
            duration_holding_min = st.number_input("홀딩 최소 지속 시간 (Hours)", value=10.0, step=0.5, help="이 시간 이상 홀딩되어야 유효한 사이클로 간주")
            temp_holding_max = st.number_input("홀딩 온도 (Max)", value=1270, step=10)
            st.write("") # 공간 맞추기
            
        # 장입 및 저온 체크 통합 (간소화)
        st.divider()
        st.subheader("⚙️ 고급 시작/종료 조건")
        check_strict_start = st.checkbox("정밀 시작/저온 체크 사용 (권장)", value=False, help="활성화 시: 1) 저온 후 승온 재시작 시점을 시작으로 포착 2) 시작 2시간 후 저온 복귀 시 사이클 제외")
        
        # 매칭 시간 허용 범위 설정 옵션 (기본값 12h -> 24h로 변경)
        time_tolerance_hours = st.number_input("생산 실적 매칭 시간 허용 범위 (Hours)", value=24, min_value=1, max_value=48, step=1)
        st.info(f"센서 사이클 시작 시각과 생산 실적의 '차지 시작 시각'이 ±{time_tolerance_hours}시간 이내일 때만 매칭됩니다.")


        st.divider()
        st.header("3. 엑셀/CSV 설정")
        # 사용자가 원하는 행을 직접 선택하는 기능 (제목행 인덱스 선택)
        p_header = st.number_input("생산실적 제목행 (0부터 시작)", 0, 10, 0)
        s_header = st.number_input("가열로 데이터 제목행 (0부터 시작)", 0, 20, 0)
        
        run_btn = st.button("🚀 분석 실행", type="primary")

    # 가열로 이름을 제목에 반영 (분석 전에는 일반적인 제목 사용)
    st.title(f"🏭 가열로 다중 분석 시스템 (최대 20개)")
    
    if prod_file and sensor_files:
        st.subheader("🛠️ 데이터 컬럼 지정 (미리보기)")
        st.warning("⚠️ **중요:** 생산 실적 데이터의 '차지 시작 시각 컬럼'은 개별 차지(작업)의 정확한 시작 시간을 포함해야 분석 정확도가 높습니다.")
        
        try:
            # 미리보기 데이터 로드 (첫 3줄)
            df_p = smart_read_file(prod_file, p_header, 3)
            prod_file.seek(0) # 파일 포인터 초기화
            
            f = sensor_files[0]; f.seek(0)
            df_s = smart_read_file(f, s_header, 3)
            f.seek(0) # 파일 포인터 초기화
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.caption("생산 실적 데이터")
                st.dataframe(df_p)
                
                # 키워드 기반 기본 인덱스 설정
                # '가열시작일시' 또는 '일시'를 우선 찾음
                col_p_start_time_index = get_default_index(df_p.columns, ['가열시작일시', '시작일시', '일시', 'date', '시간'])
                col_p_weight_index = get_default_index(df_p.columns, ['장입', '중량', 'weight', 'kg'])
                col_p_unit_index = get_default_index(df_p.columns, ['가열로', '호기', 'unit', 'furnace', '명'])
                
                # 사용자가 원하는 컬럼 이름 직접 선택
                # 날짜 컬럼 대신 '차지 시작 시각' 컬럼 선택으로 변경
                col_p_start_time = st.selectbox("⏰ 차지 시작 시각 컬럼", df_p.columns, index=col_p_start_time_index, key="p_start_time")
                col_p_weight = st.selectbox("⚖️ 장입량 컬럼", df_p.columns, index=col_p_weight_index, key="p_weight")
                col_p_unit = st.selectbox("🏭 생산 실적의 가열로 ID 컬럼", df_p.columns, index=col_p_unit_index, key="p_unit")
                
            with c2:
                st.caption("가열로 센서 데이터 (가열로 ID는 파일 이름에서 추출)")
                st.dataframe(df_s)
                
                # 키워드 기반 기본 인덱스 설정
                col_s_time_index = get_default_index(df_s.columns, ['일시', '시간', 'time'])
                col_s_temp_index = get_default_index(df_s.columns, ['온도', 'temp', '℃'])
                col_s_gas_index = get_default_index(df_s.columns, ['가스', '지침', 'gas', '누적지침'])
                
                # 사용자가 원하는 컬럼 이름 직접 선택
                col_s_time = st.selectbox("⏰ 일시 컬럼", df_s.columns, index=col_s_time_index, key="s_time")
                col_s_temp = st.selectbox("🔥 온도 컬럼", df_s.columns, index=col_s_temp_index, key="s_temp")
                col_s_gas = st.selectbox("⛽ 가스지침 컬럼", df_s.columns, index=col_s_gas_index, key="s_gas")
                
        except Exception as e:
            st.error(f"데이터 미리보기에 실패했습니다. 제목행 설정을 확인하거나 파일 형식을 점검해주세요. (세부 오류: {e})")
            col_p_start_time, col_p_weight, col_p_unit, col_s_time, col_s_temp, col_s_gas = None, None, None, None, None, None

        if run_btn and col_p_start_time: # 컬럼 선택이 완료되었을 때 실행
            with st.spinner("정밀 분석 중... (사이클 탐색 및 원단위 계산)"):
                # 전체 데이터 다시 읽기
                f_prod_full = smart_read_file(prod_file, p_header)
                
                # process_data 호출 시 check_strict_start 전달 (check_abnormal_low, check_charging_end 대신)
                res, raw, error_msg = process_data(sensor_files, f_prod_full, 
                                                   col_p_start_time, col_p_weight, col_p_unit, 
                                                   s_header, col_s_time, col_s_temp, col_s_gas, 
                                                   target_cost, temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end, check_strict_start, use_target_cost, time_tolerance_hours)
                
                if error_msg:
                     st.error(f"분석 실패: {error_msg}")
                elif res is not None and not res.empty:
                    st.session_state['res'] = res
                    st.session_state['raw'] = raw
                    # 분석된 가열로 ID 목록을 세션에 저장
                    st.session_state['unit_ids'] = res['가열로'].unique().tolist()
                    st.session_state['use_target_cost'] = use_target_cost # 세션에 저장
                    st.session_state['target_cost'] = target_cost # 세션에 저장
                    st.success(f"분석 완료! 총 {len(st.session_state['unit_ids'])}개 가열로에서 유효 사이클 {len(res)}건 발견.")
                else:
                    st.error("분석 실패 (조건에 맞는 유효 사이클 없음)")

    if 'res' in st.session_state:
        df = st.session_state['res']
        # 분석 시점의 설정값을 세션에서 가져옴
        use_target_cost = st.session_state.get('use_target_cost', False)
        target_cost = st.session_state.get('target_cost')
        
        st.divider()
        
        # 가열로별 분석 결과를 필터링하기 위한 selectbox
        selected_unit = st.selectbox("개별 가열로 선택 (종합 통계 및 리포트 대상):", ['전체'] + st.session_state['unit_ids'], key='unit_filter')
        
        if selected_unit != '전체':
            df_filtered = df[df['가열로'] == selected_unit].copy()
        else:
            df_filtered = df.copy()
            
        t1, t2, t3 = st.tabs(["📊 분석 결과", "📈 종합 통계", "📑 리포트"])
        
        with t1:
            st.subheader(f"{selected_unit} 유효 사이클별 원단위 상세")
            # 목표 원단위를 사용하는 경우에만 Pass/Fail 색상 적용
            if use_target_cost:
                st.dataframe(df_filtered.style.applymap(lambda x: 'background-color:#d4edda; color:#155724' if x=='Pass' else 'background-color:#f8d7da; color:#721c24', subset=['달성여부']), use_container_width=True)
            else:
                st.dataframe(df_filtered, use_container_width=True)

        with t2:
            st.subheader(f"{selected_unit} 원단위 분포 및 추세 분석")
            if not df_filtered.empty:
                avg_unit = df_filtered['원단위'].mean()
                
                col_s1, col_s2, col_s3 = st.columns(3)
                if selected_unit == '전체':
                    # 모든 가열로 비교 통계
                    df_summary = df.groupby('가열로').agg(
                        총사이클=('원단위', 'size'),
                        평균원단위=('원단위', 'mean'),
                        총장입량=('장입량(kg)', 'sum'),
                        총가스사용량=('가스사용량(Nm3)', 'sum')
                    ).reset_index()
                    
                    df_summary['평균원단위'] = df_summary['평균원단위'].round(2)
                    
                    st.subheader("🔥 가열로별 평균 원단위 비교")
                    
                    # Bar Chart
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 대표 색상
                    num_units = len(df_summary)
                    
                    bars = ax_bar.bar(df_summary['가열로'], df_summary['평균원단위'], color=colors[:num_units])
                    ax_bar.set_title('가열로별 평균 원단위 (Nm3/ton)')
                    ax_bar.set_ylabel('평균 원단위')
                    ax_bar.tick_params(axis='x', rotation=45)
                    
                    if use_target_cost and target_cost is not None:
                        ax_bar.axhline(target_cost, color='r', linestyle='--', linewidth=2, label=f'목표 ({target_cost:.2f})')
                        ax_bar.legend()
                        
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)
                    
                    st.subheader("종합 요약 테이블")
                    st.dataframe(df_summary, use_container_width=True)


                else:
                    # 개별 가열로 통계 (기존 로직 유지)
                    if use_target_cost:
                        pass_count = (df_filtered['달성여부'] == 'Pass').sum()
                        fail_count = (df_filtered['달성여부'] == 'Fail').sum()
                        with col_s1: st.metric("평균 원단위", f"{avg_unit:.2f} Nm3/ton", f"{avg_unit - target_cost:.2f}", delta_color="inverse")
                        with col_s2: st.metric("Pass 건수", f"{pass_count} 건")
                        with col_s3: st.metric("Fail 건수", f"{fail_count} 건")
                    else:
                        with col_s1: st.metric("평균 원단위", f"{avg_unit:.2f} Nm3/ton")
                        with col_s2: st.metric("총 사이클", f"{len(df_filtered)} 건")
                        with col_s3: st.write("")

                    # 1. 히스토그램 (분포)
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                    df_filtered['원단위'].hist(ax=ax_hist, bins=15, edgecolor='black', alpha=0.7)
                    
                    if use_target_cost:
                        ax_hist.axvline(target_cost, color='r', linestyle='--', linewidth=2, label=f'목표 ({target_cost:.2f})')
                    
                    ax_hist.axvline(avg_unit, color='g', linestyle='-', linewidth=2, label=f'평균 ({avg_unit:.2f})')
                    ax_hist.set_title(f'[{selected_unit}] 원단위 분포 히스토그램')
                    ax_hist.set_xlabel('원단위 (Nm3/ton)')
                    ax_hist.set_ylabel('사이클 수')
                    ax_hist.legend()
                    st.pyplot(fig_hist)
                    plt.close(fig_hist) # 메모리 해제
                    
                    # 2. 시계열 차트 (추세)
                    fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
                    df_trend = df_filtered.copy()
                    df_trend['날짜'] = pd.to_datetime(df_trend['날짜'])

                    ax_trend.plot(df_trend['날짜'], df_trend['원단위'], marker='o', linestyle='-', color='b', label='실적 원단위')
                    
                    if use_target_cost:
                        ax_trend.axhline(target_cost, color='r', linestyle='--', linewidth=2, label=f'목표 ({target_cost:.2f})')
                    
                    ax_trend.set_title(f'[{selected_unit}] 원단위 시계열 추이')
                    ax_trend.set_xlabel('날짜')
                    ax_trend.set_ylabel('원단위 (Nm3/ton)')
                    ax_trend.legend()
                    ax_trend.grid(True, linestyle=':', alpha=0.6)
                    st.pyplot(fig_trend)
                    plt.close(fig_trend) # 메모리 해제
            else:
                 st.warning("분석할 유효 데이터가 없습니다.")

        with t3:
            # 리포트 생성 조건 설정
            can_generate_report = False
            if selected_unit == '전체':
                st.warning("리포트는 개별 가열로를 선택했을 때만 생성이 가능합니다.")
            elif df_filtered.empty:
                 st.warning(f"가열로 {selected_unit}의 분석 데이터가 없어 리포트 생성이 불가합니다.")
            elif use_target_cost:
                df_pass = df_filtered[df_filtered['달성여부'] == 'Pass']
                if df_pass.empty:
                    st.warning(f"가열로 {selected_unit}의 목표 달성 데이터가 없어 리포트 생성이 불가합니다. (목표 원단위 사용 중)")
                else:
                    can_generate_report = True
            else: # 목표 원단위를 사용하지 않는 경우, 모든 사이클을 리포트 대상으로 간주
                df_pass = df_filtered.copy()
                can_generate_report = True

            if can_generate_report:
                s_date = st.selectbox("리포트 생성 대상 날짜 선택:", df_pass['날짜'].unique(), key='report_date')
                
                row = df_pass[df_pass['날짜'] == s_date].iloc[0]
                
                # --- 차트 미리보기: 날짜 선택 시 바로 표시 ---
                st.subheader("▶️ 열처리 Chart 미리보기 (온도/가스 트렌드)")
                
                # plot_cycle_chart 호출하여 fig 생성 (미리보기 크기 10x5)
                fig_preview = plot_cycle_chart(row, st.session_state['raw'], temp_holding_min, temp_holding_max, fig_width=10, fig_height=5)
                st.pyplot(fig_preview)
                plt.close(fig_preview) # 메모리 해제
                
                # --- PDF 생성 버튼 ---
                if st.button("PDF 리포트 생성", key='generate_pdf_button'):
                    with st.spinner("리포트 및 차트 생성 중..."):
                        # PDF용 차트 (리포트용 크기 12x5)
                        fig_pdf = plot_cycle_chart(row, st.session_state['raw'], temp_holding_min, temp_holding_max, fig_width=12, fig_height=5)
                        
                        # 임시 파일에 저장
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            fig_pdf.savefig(tmp.name, bbox_inches='tight')
                            img_path = tmp.name
                        
                        plt.close(fig_pdf)
                        
                        try:
                            # unit_name, use_target_cost, target_cost를 generate_pdf로 전달
                            pdf = generate_pdf(row, img_path, target_cost, selected_unit, use_target_cost)
                            pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            st.download_button("📥 다운로드", pdf_bytes, f"Report_{selected_unit}_{s_date}.pdf", "application/pdf")
                        finally:
                            os.remove(img_path)
                        
                        st.success(f"PDF 리포트가 생성되었습니다. ({s_date})")

if __name__ == "__main__":
    main()
