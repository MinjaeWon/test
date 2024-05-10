import openai
import streamlit as st
import pandas as pd
import re
import altair as alt


# 와이드 레이아웃 설정
st.set_page_config(layout="wide", page_title="메뉴 영양소 분석 도구")

# OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 필수 영양소 7가지에 대한 권장량
essential_nutrients = {
    '에너지': 2600, '단백질': 65, '지방': 58, '탄수화물': 357, '비타민 C': 100,
    '칼슘': 800, '철': 10
}

# 모든 영양소 권장량 (기존에 정의한 것 유지)
recommended_values = {
    '에너지': 2600, '단백질': 65, '지방': 58, '탄수화물': 357, '식이섬유': 25, '비타민 A': 750,
    '비타민 C': 100, '비타민 D': 10, '비타민 E': 12, '비타민 K': 75, '티아민': 1.2,
    '리보플라빈': 1.5, '나이아신': 16, '비타민 B6': 1.5, '엽산': 400, '비타민 B12': 2.4,
    '판토텐산': 5, '비오틴': 30, '칼슘': 800, '인': 700, '나트륨': 1500, '칼륨': 3500,
    '마그네슘': 350, '철': 10, '아연': 10, '구리': 900, '셀레늄': 60, '망간': 4
}

# 사이드바에 "메뉴 분석 도구" 문구만 표시
with st.sidebar:
    st.title("메뉴 분석 도구")

# GPT에게 메뉴 분석 요청
def ask_gpt_for_nutrition_analysis(menu_list, recommended_values):
    menu_items = ', '.join(menu_list)

    # 권장량을 GPT에 전달하기 위해 문자열로 변환
    recommended_str = '\n'.join([f"{k}: {v}" for k, v in recommended_values.items()])

    prompt = f"""
    입력된 메뉴는 {menu_items}입니다. 한국인 20대 남성의 영양소 권장량은 다음과 같습니다:
    {recommended_str}

    입력된 메뉴의 영양소를 분석하고 권장량과 비교한 결과를 숫자와 상태로 표시해 주세요.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 영양 전문가입니다. 메뉴의 영양소를 권장량과 비교해 분석해 주세요."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )
    return response['choices'][0]['message']['content']

# 영양소 데이터를 수집하는 함수 (GPT 응답 파싱)
def parse_nutrition_data(gpt_response):
    pattern = re.compile(r"(\w+):\s*(\d+\.?\d*)")
    matches = pattern.findall(gpt_response)

    nutrition_dict = {'영양소': [], '섭취량': [], '권장량': []}
    
    for nutrient, value in matches:
        nutrition_dict['영양소'].append(nutrient)
        nutrition_dict['섭취량'].append(float(value))
        nutrition_dict['권장량'].append(recommended_values.get(nutrient, 0))

    return pd.DataFrame(nutrition_dict)

# 오른쪽 화면을 col1과 col2로 구분
col1, col2 = st.columns(2)

# col1에 조식, 중식, 석식별로 메뉴 입력 필드를 생성
with col1:
    st.subheader("메뉴 분석 도구")
    st.write("**조식, 중식, 석식 메뉴**를 각각 입력하세요 (콤마로 구분):")
    breakfast_input = st.text_input("조식 메뉴 👇")
    lunch_input = st.text_input("중식 메뉴 👇")
    dinner_input = st.text_input("석식 메뉴 👇")
    analyze_button = st.button("분석")

    # 모든 입력된 메뉴를 통합하여 분석에 사용
    all_menus = [menu.strip() for menu in f"{breakfast_input}, {lunch_input}, {dinner_input}".split(',') if menu.strip()]

    # 분석 결과 초기화
    analysis_result = ""
    nutrition_data = pd.DataFrame()

    # 분석 버튼을 누른 경우 GPT 분석 결과 표시
    if analyze_button and all_menus:
        analysis_result = ask_gpt_for_nutrition_analysis(all_menus, recommended_values)
        nutrition_data = parse_nutrition_data(analysis_result)
    else:
        analysis_result = "오류: 메뉴가 입력되지 않았습니다."

    st.text_area("영양소 분석 결과", value=analysis_result, height=200)

# col2에 필수 영양소 7가지를 시각화 및 상태표 표시
with col2:
    st.subheader("필수 영양소 7가지 차트 및 상태표")
    if not nutrition_data.empty:
        # 필수 영양소 데이터 필터링
        essential_df = nutrition_data[nutrition_data['영양소'].isin(essential_nutrients.keys())]

        # 데이터 변환 및 꺾은선 그래프 생성
        chart_data = essential_df.melt(id_vars='영양소', value_vars=['섭취량', '권장량'], var_name='종류', value_name='값')
        line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X('영양소:N', sort=essential_df['영양소'].tolist()),
            y='값:Q',
            color='종류:N'
        ).properties(width=600, height=400, title="필수 영양소 섭취량과 권장량 비교 꺾은선 그래프")

        st.altair_chart(line_chart)

        

        # 필수 영양소 상태표를 만들기 위한 함수
        def determine_status(row):
            nutrient = row['영양소']
            intake = row['섭취량']
            recommended = essential_nutrients.get(nutrient, 0)
            if intake >= recommended * 0.8 and intake <= recommended * 1.2:
                return '적절'
            elif intake < recommended * 0.8:
                return '부족'
            else:
                return '과잉'

        # 상태 계산 및 추가
        essential_df['상태'] = essential_df.apply(determine_status, axis=1)


        # 필수 영양소 상태표를 표 형태로 표시
        st.table(essential_df[['영양소', '섭취량', '권장량', '상태']])
    else:
        st.write("분석된 영양소 데이터가 없습니다.")
