import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF
import tempfile
import os
from datetime import timedelta
import numpy as np

# ---------------------------------------------------------
# 1. 앱 설정 및 폰트
# ---------------------------------------------------------
st.set_page_config(page_title="가열로 5호기 정밀 분석", layout="wide")

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
def analyze_cycle(daily_data, temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end):
    """
    조건:
    1. 시작: temp_start 이하
    2. 홀딩: temp_holding_min ~ temp_holding_max 구간이 duration_holding_min 이상 지속
    3. 종료: 홀딩 이후 temp_end 이하로 떨어지는 시점
    4. 유효성: 시작 2시간 후부터 종료 시점까지 temp_start 미만으로 떨어지지 않아야 함 (수정된 로직)
    """
    # 1. 시작점 찾기
    start_candidates = daily_data[daily_data['온도'] <= temp_start]
    if start_candidates.empty:
        return None, f"시작 온도({temp_start}도 이하) 없음"
    start_row = start_candidates.iloc[0]
    start_time = start_row['일시']

    # 2. 홀딩 구간 찾기
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
        return None, f"유효 홀딩 구간({duration_holding_min}시간 이상) 없음"

    # 3. 종료점 찾기
    post_holding_data = daily_data[daily_data['일시'] > holding_end_time]
    end_candidates = post_holding_data[post_holding_data['온도'] <= temp_end]
    
    if end_candidates.empty:
        return None, f"종료 온도({temp_end}도 이하) 도달 안 함"
        
    end_row = end_candidates.iloc[0]
    end_time = end_row['일시']

    # 4. 사이클 시작 후 2시간 이후에 비정상적인 저온 발생 여부 확인 (수정된 로직)
    
    # 2시간 후의 시작 시점 정의
    check_start_time = start_time + timedelta(hours=2)
    
    # 체크 윈도우: 시작 2시간 후부터 종료 시간 직전까지의 데이터 추출
    cycle_window = daily_data[(daily_data['일시'] >= check_start_time) & (daily_data['일시'] < end_time)].copy()

    # 이 구간 내에서 시작 온도(temp_start)보다 엄격하게 낮은 온도가 있는지 확인
    abnormal_low_temp = cycle_window[cycle_window['온도'] < temp_start]
    
    if not abnormal_low_temp.empty:
        abnormal_time = abnormal_low_temp.iloc[0]['일시'].strftime('%Y-%m-%d %H:%M')
        return None, f"사이클 시작 2시간 후 비정상적인 저온 발생 (<{temp_start}℃) at {abnormal_time}"
    # (수정 로직 종료)
    
    return {
        'start_row': start_row,
        'end_row': end_row,
        'holding_end': holding_end_time
    }, "성공"

def process_data(sensor_files, df_prod, col_p_date, col_p_weight, 
                 s_header_row, col_s_time, col_s_temp, col_s_gas, target_cost, 
                 temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end):
    
    # --- 생산실적 데이터 전처리 ---
    try:
        df_prod = df_prod.rename(columns={col_p_date: '일자', col_p_weight: '장입량'})
        df_prod['일자'] = pd.to_datetime(df_prod['일자'], errors='coerce').dt.normalize() # 시간 제거
        if df_prod['장입량'].dtype == object:
            df_prod['장입량'] = df_prod['장입량'].astype(str).str.replace(',', '')
        df_prod['장입량'] = pd.to_numeric(df_prod['장입량'], errors='coerce')
        df_prod = df_prod.dropna(subset=['일자', '장입량']).sort_values('일자')
    except Exception as e: return None, None, f"생산실적 오류: {e}"

    # --- 센서 데이터 통합 및 전처리 ---
    df_list = []
    for f in sensor_files:
        f.seek(0) # 파일 포인터 초기화
        df = smart_read_file(f, s_header_row)
        if df is not None: df_list.append(df)
    
    if not df_list: return None, None, "센서 데이터 없음"
    
    df_sensor = pd.concat(df_list, ignore_index=True)
    df_sensor.columns = [str(c).strip() for c in df_sensor.columns]
    
    try:
        df_sensor = df_sensor.rename(columns={col_s_time: '일시', col_s_temp: '온도', col_s_gas: '가스지침'})
        df_sensor['일시'] = pd.to_datetime(df_sensor['일시'], errors='coerce')
        df_sensor['온도'] = pd.to_numeric(df_sensor['온도'], errors='coerce')
        df_sensor['가스지침'] = pd.to_numeric(df_sensor['가스지침'], errors='coerce')
        # 시간 컬럼 기준으로 정렬하고 NaN 제거
        df_sensor = df_sensor.dropna(subset=['일시']).sort_values('일시')
        # 중복 일시 제거 (가장 마지막 값 유지)
        df_sensor = df_sensor.drop_duplicates(subset=['일시'], keep='last').reset_index(drop=True)
    except Exception as e: return None, None, f"센서 데이터 매핑 오류: {e}"

    # --- 분석 실행 ---
    # 생산실적 날짜 기준으로 분석 (하루의 사이클은 24시간을 넘길 수 있으므로 48시간 윈도우 사용)
    prod_dates = df_prod['일자'].dt.normalize().unique()
    
    if len(prod_dates) == 0: return None, None, "날짜 매칭 실패: 유효한 생산실적 날짜 없음"

    results = []
    
    for date_ts in prod_dates:
        date = date_ts.date()
        prod_row = df_prod[df_prod['일자'] == date_ts].iloc[0]
        
        # 48시간 윈도우 데이터
        daily_window = df_sensor[
            (df_sensor['일시'] >= date_ts - timedelta(hours=1)) & # 하루 전부터 시작해서 혹시 모를 사이클 시작점 포함
            (df_sensor['일시'] < date_ts + timedelta(days=2)) # 다음날 끝까지
        ].copy()
        
        if daily_window.empty: continue
        
        # 사이클 분석 수행
        cycle_info, msg = analyze_cycle(daily_window, temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end)
        
        if cycle_info:
            start = cycle_info['start_row']
            end = cycle_info['end_row']
            
            charge_kg = prod_row['장입량']
            
            # 장입량 또는 가스 사용량이 비정상이면 건너뛰기
            if charge_kg <= 0: continue
            gas_used = end['가스지침'] - start['가스지침']
            if gas_used <= 0: continue
            
            unit = gas_used / (charge_kg / 1000) # Nm3 / ton
            # 목표 원단위보다 작거나 같을 때 'Pass'
            is_pass = unit <= target_cost
            
            results.append({
                '날짜': date.strftime('%Y-%m-%d'),
                '검침시작': start['일시'].strftime('%Y-%m-%d %H:%M'),
                '시작지침': start['가스지침'],
                '검침완료': end['일시'].strftime('%Y-%m-%d %H:%M'),
                '종료지침': end['가스지침'],
                '가스사용량(Nm3)': int(gas_used),
                '장입량(kg)': int(charge_kg),
                '원단위': round(unit, 2),
                '달성여부': 'Pass' if is_pass else 'Fail',
                '비고': f"홀딩종료: {cycle_info['holding_end'].strftime('%H:%M')}"
            })
            
    return pd.DataFrame(results), df_sensor, None

# ---------------------------------------------------------
# 4. PDF 생성
# ---------------------------------------------------------
class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if HAS_KOREAN_FONT: self.add_font('Nanum', '', FONT_FILE, uni=True)

    def header(self):
        font = 'Nanum' if HAS_KOREAN_FONT else 'Arial'
        self.set_font(font, 'B' if not HAS_KOREAN_FONT else '', 14)
        self.cell(0, 10, '3. 가열로 5호기 검증 DATA (개선 후)', 0, 1, 'L')
        self.ln(5)

def generate_pdf(row_data, chart_path, target):
    pdf = PDFReport()
    pdf.add_page()
    font = 'Nanum' if HAS_KOREAN_FONT else 'Arial'
    
    pdf.set_font(font, '', 12)
    pdf.cell(0, 10, f"3.5 가열로 5호기 - {row_data['날짜']} (23% 절감 검증)", 0, 1, 'L')
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
    pdf.cell(0, 8, f"* 실적 원단위: {row_data['원단위']} Nm3/ton (목표 {target} 이하 달성)", 0, 1, 'R')
    
    return pdf

# ---------------------------------------------------------
# 4.5 차트 생성 함수 (미리보기 및 PDF용)
# ---------------------------------------------------------
def plot_cycle_chart(row, full_raw, temp_holding_min, temp_holding_max, fig_width=10, fig_height=5):
    """주어진 사이클 정보를 바탕으로 Matplotlib 차트를 생성하여 반환합니다."""
    s_ts = pd.to_datetime(row['검침시작'])
    e_ts = pd.to_datetime(row['검침완료'])
    
    # 앞뒤로 1시간 여유 두기
    chart_data = full_raw[(full_raw['일시'] >= s_ts - timedelta(hours=1)) & (full_raw['일시'] <= e_ts + timedelta(hours=1))].copy()
    
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
    
    plt.title(f"Cycle: {row['검침시작']} ~ {row['검침완료']}")
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
    st.title("🏭 가열로 5호기 정밀 검증 시스템")
    
    with st.sidebar:
        st.header("1. 데이터 업로드")
        prod_file = st.file_uploader("생산 실적 (Excel)", type=['xlsx'])
        sensor_files = st.file_uploader("가열로 데이터 (CSV/Excel)", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)
        
        st.divider()
        st.header("2. 분석 기준 설정")
        target_cost = st.number_input("목표 원단위 (Nm3/ton)", value=25.53, step=0.1, format="%.2f")
        
        st.subheader("🔥 사이클 정의 (온도/시간)")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            temp_start = st.number_input("시작 온도 (Max)", value=600, step=10)
            temp_holding_min = st.number_input("홀딩 온도 (Min)", value=1230, step=10)
            temp_end = st.number_input("종료 온도 (Max)", value=900, step=10)
        with col_t2:
            duration_holding_min = st.number_input("홀딩 최소 지속 시간 (Hours)", value=10.0, step=0.5)
            temp_holding_max = st.number_input("홀딩 온도 (Max)", value=1270, step=10)
            st.write("")
            
        st.info(f"기준: Start < {temp_start}℃, {duration_holding_min}hr Holding ({temp_holding_min}~{temp_holding_max}℃), End < {temp_end}℃")
        
        st.divider()
        st.header("3. 엑셀/CSV 설정")
        # 사용자가 원하는 행을 직접 선택하는 기능 (제목행 인덱스 선택)
        p_header = st.number_input("생산실적 제목행 (0부터 시작)", 0, 10, 0)
        s_header = st.number_input("가열로 데이터 제목행 (0부터 시작)", 0, 20, 0)
        
        run_btn = st.button("🚀 분석 실행", type="primary")

    if prod_file and sensor_files:
        st.subheader("🛠️ 데이터 컬럼 지정 (미리보기)")
        
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
                col_p_date_index = get_default_index(df_p.columns, ['날짜', '일자', 'date'])
                col_p_weight_index = get_default_index(df_p.columns, ['장입', '중량', 'weight'])
                
                # 사용자가 원하는 컬럼 이름 직접 선택
                col_p_date = st.selectbox("📅 날짜 컬럼", df_p.columns, index=col_p_date_index, key="p_date")
                col_p_weight = st.selectbox("⚖️ 장입량 컬럼", df_p.columns, index=col_p_weight_index, key="p_weight")
                
            with c2:
                st.caption("가열로 센서 데이터")
                st.dataframe(df_s)
                
                # 키워드 기반 기본 인덱스 설정
                col_s_time_index = get_default_index(df_s.columns, ['일시', '시간', 'time'])
                col_s_temp_index = get_default_index(df_s.columns, ['온도', 'temp'])
                col_s_gas_index = get_default_index(df_s.columns, ['가스', '지침', 'gas'])
                
                # 사용자가 원하는 컬럼 이름 직접 선택
                col_s_time = st.selectbox("⏰ 일시 컬럼", df_s.columns, index=col_s_time_index, key="s_time")
                col_s_temp = st.selectbox("🔥 온도 컬럼", df_s.columns, index=col_s_temp_index, key="s_temp")
                col_s_gas = st.selectbox("⛽ 가스지침 컬럼", df_s.columns, index=col_s_gas_index, key="s_gas")
                
        except Exception as e:
            st.error(f"데이터 미리보기에 실패했습니다. 제목행 설정을 확인하거나 파일 형식을 점검해주세요. (세부 오류: {e})")
            col_p_date, col_p_weight, col_s_time, col_s_temp, col_s_gas = None, None, None, None, None

        if run_btn and col_p_date: # 컬럼 선택이 완료되었을 때 실행
            with st.spinner("정밀 분석 중... (사이클 탐색 및 원단위 계산)"):
                # 전체 데이터 다시 읽기
                f_prod_full = smart_read_file(prod_file, p_header)
                
                res, raw, error_msg = process_data(sensor_files, f_prod_full, 
                                                   col_p_date, col_p_weight, 
                                                   s_header, col_s_time, col_s_temp, col_s_gas,
                                                   target_cost, temp_start, temp_holding_min, temp_holding_max, duration_holding_min, temp_end)
                
                if error_msg:
                     st.error(f"분석 실패: {error_msg}")
                elif res is not None and not res.empty:
                    st.session_state['res'] = res
                    st.session_state['raw'] = raw
                    st.success(f"분석 완료! 유효 사이클 {len(res)}건 발견.")
                else:
                    st.error("분석 실패 (조건에 맞는 유효 사이클 없음)")

    if 'res' in st.session_state:
        df = st.session_state['res']
        st.divider()
        t1, t2, t3 = st.tabs(["📊 분석 결과", "📈 종합 통계", "📑 리포트"])
        
        with t1:
            st.subheader("유효 사이클별 원단위 상세")
            st.dataframe(df.style.applymap(lambda x: 'background-color:#d4edda; color:#155724' if x=='Pass' else 'background-color:#f8d7da; color:#721c24', subset=['달성여부']), use_container_width=True)
            
        with t2:
            st.subheader("원단위 분포 및 추세 분석")
            if not df.empty:
                avg_unit = df['원단위'].mean()
                pass_count = (df['달성여부'] == 'Pass').sum()
                fail_count = (df['달성여부'] == 'Fail').sum()
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1: st.metric("평균 원단위", f"{avg_unit:.2f} Nm3/ton", f"{avg_unit - target_cost:.2f}", delta_color="inverse")
                with col_s2: st.metric("Pass 건수", f"{pass_count} 건")
                with col_s3: st.metric("Fail 건수", f"{fail_count} 건")

                # 1. 히스토그램 (분포)
                fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                df['원단위'].hist(ax=ax_hist, bins=15, edgecolor='black', alpha=0.7)
                ax_hist.axvline(target_cost, color='r', linestyle='--', linewidth=2, label=f'목표 ({target_cost:.2f})')
                ax_hist.axvline(avg_unit, color='g', linestyle='-', linewidth=2, label=f'평균 ({avg_unit:.2f})')
                ax_hist.set_title('원단위 분포 히스토그램')
                ax_hist.set_xlabel('원단위 (Nm3/ton)')
                ax_hist.set_ylabel('사이클 수')
                ax_hist.legend()
                st.pyplot(fig_hist)
                plt.close(fig_hist) # 메모리 해제
                
                # 2. 시계열 차트 (추세)
                df_trend = df.copy()
                df_trend['날짜'] = pd.to_datetime(df_trend['날짜'])
                
                fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
                ax_trend.plot(df_trend['날짜'], df_trend['원단위'], marker='o', linestyle='-', color='b', label='실적 원단위')
                ax_trend.axhline(target_cost, color='r', linestyle='--', linewidth=2, label=f'목표 ({target_cost:.2f})')
                ax_trend.set_title('원단위 시계열 추이')
                ax_trend.set_xlabel('날짜')
                ax_trend.set_ylabel('원단위 (Nm3/ton)')
                ax_trend.legend()
                ax_trend.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig_trend)
                plt.close(fig_trend) # 메모리 해제
            else:
                 st.warning("분석할 유효 데이터가 없습니다.")

        with t3:
            df_pass = df[df['달성여부'] == 'Pass']
            if df_pass.empty:
                st.warning("목표 원단위를 달성한 데이터가 없어 리포트 생성이 불가합니다.")
            else:
                s_date = st.selectbox("리포트 생성 대상 날짜 선택:", df_pass['날짜'].unique(), key='report_date')
                
                row = df_pass[df_pass['날짜'] == s_date].iloc[0]
                
                # --- 차트 미리보기: 날짜 선택 시 바로 표시 ---
                st.subheader("▶️ 열처리 Chart 미리보기 (온도/가스 트렌드)")
                
                # plot_cycle_chart 호출하여 fig 생성 (미리보기 크기 10x5)
                # 사이클 정의 파라미터는 main() 함수 스코프에서 가져옵니다.
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
                            pdf = generate_pdf(row, img_path, target_cost)
                            pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            st.download_button("📥 다운로드", pdf_bytes, f"Report_{s_date}.pdf", "application/pdf")
                        finally:
                            os.remove(img_path)
                        
                        st.success(f"PDF 리포트가 생성되었습니다. ({s_date})")

if __name__ == "__main__":
    main()