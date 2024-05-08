import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 모델 및 데이터 로드
import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 모델 및 데이터 로드
@st.cache_data
def load_data():
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    df = pd.read_csv('https://raw.githubusercontent.com/MinjaeWon/test/main/simri_dataset.csv', encoding='utf-8')
    df['embedding'] = df['embedding'].apply(json.loads)
    return model, df

model, df = load_data()

# OpenAI API Key 설정
openai.api_key = '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' # 나의 API 키

def generate_response(text):
    embedding = model.encode(text)
    df['distance'] = df['embedding'].apply(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    return answer

#GPT로 대답하기.
def generate_gpt_response(text):
    # 초기 설정에 "You are a helpful and empathetic assistant" 추가
    # init_prompt = """
    # You are a helpful and empathetic assistant trained as a psychological expert. You engage in deep conversations by empathetically responding and asking open-ended questions to encourage the user to share more about their feelings and situation.
    #  """
    init_prompt = """
    You are a helpful and empathetic assistant trained as a psychological expert. Your goal is to support meaningful and deep conversations. You achieve this by expressing empathy in response to the user's inputs and by asking open-ended questions that encourage them to elaborate on their feelings and situation.  """
    messages = [
        {"role": "system", "content": init_prompt},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,  # 토큰 수 증가로 더 많은 내용을 포함할 수 있도록 조정
        temperature=0.8,
        stop=["\n", ".", "!", "?"]  # 문장 종료 시점을 조절
    )
    # 응답 중 종결 구두점을 찾고 그 지점까지의 내용을 반환
    full_response = response['choices'][0]['message']['content'].strip()
    end_punctuations = {".", "?", "!"}
    for punct in end_punctuations:
        punct_index = full_response.find(punct)
        if punct_index != -1:
            full_response = full_response[:punct_index + 1]
            break
    return full_response

# #학교폭력 의심 지수 측정
def check_bullying(user_input):
    # 사용자 입력의 임베딩 계산
    print("---user input", user_input )
    user_embedding = model.encode(user_input)
    # 사전 임베딩과의 유사도 계산
    df['similarity'] = df['embedding'].apply(
        lambda emb: cosine_similarity([user_embedding], [emb]).squeeze()
    )
    # 평균 유사도 계산
    average_similarity = df['similarity'].mean()
    threshold = 0  # 유사도 임계값 설정
    # 유사도 평균이 임계값을 초과하면 학교폭력 가능성을 판단
    print("뉴 학교 폭력 측정 지수 :", average_similarity)
    return average_similarity > threshold, average_similarity

bullying_keywords = {
    "학교폭력": 1,
    "괴롭힘": 1,
    "협박": 1,
    "위협": 1,
    "모욕": 1,
    "폭행": 1,
    "공격": 1,
    "가해": 1,
    "피해": 1,
    "목표": 1,
    "조롱": 1,
    "따돌림": 1,
    "포함제외": 1,
    "사회적편견": 1,
    "피해자": 1,
    "가해자": 1,
    "선생님괴롭힘": 1,
    "억지로쓰는글쓰기": 1,
    "협박하다": 1,
    "공격적": 1,
    "무력": 1,
    "폭력": 1,
    "억압": 1,
    "위협적": 1,
    "무서움": 1,
    "공포": 1
}

#GPT로 자세히 말하기 (지금 안써)
def generate_detailed_feedback(text, bullying_details):
    context = "Analyzing the provided text for indications of bullying based on keyword similarities: "
    for keyword, similarity in bullying_details.items():
        context += f"{keyword}: {similarity:.2f}, "
    context = context.rstrip(", ")  # 마지막 콤마 제거
    #prompt = f"{context}. 너의 역할은 심리상담 전문가야. 학교폭력을 당하고 있는 친구가 사용자일 확률이 높아. 이 분석을 기반으로 상황이 학교폭력과 관련되어 있는지에 대해 자세한 설명을 해줘. 학교폭력 가능성이 있다면, 주변에 도움을 받으라고 조언해줘. max_tokens 300을 넘기지마."
    prompt = f"{context}. 너의 역할은 심리상담 전문가야. 입력값 기반으로 상황이 학교폭력과 관련되어 있는지에 대해 전문성 있는 느낌 나게 분석 해줘. 최대 5문장으로 말해줘."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes texts for bullying."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=650
    )
    return response['choices'][0]['message']['content']



#대화 기록을 저장하기 위한 초기 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 대화 엔진 초기화 (가상의 함수, 실제 구현 필요)
def initialize_conversation():
    return None  # 여기에 실제 대화 엔진 초기화 로직 구현

if 'conversation' not in st.session_state:
    st.session_state.conversation = initialize_conversation()

# 사용자 입력 받기
query = st.chat_input("요즘 어떠세요. 하고 싶은말을 자유롭게 해보세요.")
if query:
    # 사용자 입력을 대화 기록에 저장
    st.session_state.history.append({"role": "user", "content": query})

    # 스피너를 사용하여 로딩 중 표시
    #with st.spinner("잠시만 기다려주세요..."):
    # 대화 엔진 또는 챗봇 모델로부터 응답 받기
    answer = generate_response(query)  # 가상의 응답 생성 함수

    # 응답 로직 분기
    print("--1--",answer )
    if answer['distance'] < 0.15:
        # 유사도가 낮으면 GPT 모델을 사용하여 응답 생성
        response = generate_gpt_response(query)
    else:
        # 유사도가 높은 경우, 기존 챗봇 응답 사용
        response = f"{answer['챗봇']}."

        # 챗봇 응답을 대화 기록에 저장
    st.session_state.history.append({"role": "assistant", "content": response})

# 대화 내용 순차적으로 출력
for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


#- 윗부분 메인페이지와 겹쳐서 막음(4/25)






# 사이드 바에 Streamlit UI
with st.sidebar:
    if st.sidebar.button("폭력 피해 탐지"):
        texts = [msg['content'] for msg in st.session_state.history if msg['role'] == 'user']
        bullying_results = [check_bullying(text, model) for text in texts]
        is_bullying_present = any(result[0] for result in bullying_results)  # 유사도 결과 검토
        average_indices = [result[1] for result in bullying_results]
        average_index = sum(average_indices) / len(average_indices) if average_indices else 0

        # 평균 유사도에 따른 5단계 결과 메시지 출력
        #print(average_index)
        average_percentage = round(average_index * 10000)
        print("--", average_percentage)
        st.write(f" * AI분석 통한 학교폭력 피해 가능성: {average_percentage}%")
        if 0 <= average_percentage <= 25:
           st.success("학교 폭력 피해 가능성 비교적 낮습니다.")
        elif 26 <= average_percentage <= 50:
           st.info("학교 폭력 피해 가능성이 보통이상으로 추가 상담이 필요합니다.")
        elif 51 <= average_percentage <= 100:
           st.error("학교 폭력 피해 가능성이 비교적 높은편으로 보호자의 관리가 필요합니다.")

        print("------------------------")
        # generate_detailed_feedback 함수 호출
        detailed_feedback = generate_detailed_feedback(texts, bullying_keywords)
        print("상세 피드백:")
        print(detailed_feedback)
    

# OpenAI API Key 설정
openai.api_key = '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' # 나의 API 키

def generate_response(text):
    embedding = model.encode(text)
    df['distance'] = df['embedding'].apply(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    return answer

#GPT로 대답하기.
def generate_gpt_response(text):
    # 초기 설정에 "You are a helpful and empathetic assistant" 추가
    # init_prompt = """
    # You are a helpful and empathetic assistant trained as a psychological expert. You engage in deep conversations by empathetically responding and asking open-ended questions to encourage the user to share more about their feelings and situation.
    #  """
    init_prompt = """
    You are a helpful and empathetic assistant trained as a psychological expert. Your goal is to support meaningful and deep conversations. You achieve this by expressing empathy in response to the user's inputs and by asking open-ended questions that encourage them to elaborate on their feelings and situation.  """
    messages = [
        {"role": "system", "content": init_prompt},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,  # 토큰 수 증가로 더 많은 내용을 포함할 수 있도록 조정
        temperature=0.8,
        stop=["\n", ".", "!", "?"]  # 문장 종료 시점을 조절
    )
    # 응답 중 종결 구두점을 찾고 그 지점까지의 내용을 반환
    full_response = response['choices'][0]['message']['content'].strip()
    end_punctuations = {".", "?", "!"}
    for punct in end_punctuations:
        punct_index = full_response.find(punct)
        if punct_index != -1:
            full_response = full_response[:punct_index + 1]
            break
    return full_response

# #학교폭력 의심 지수 측정
def check_bullying(user_input):
    # 사용자 입력의 임베딩 계산
    print("---user input", user_input )
    user_embedding = model.encode(user_input)
    # 사전 임베딩과의 유사도 계산
    df['similarity'] = df['embedding'].apply(
        lambda emb: cosine_similarity([user_embedding], [emb]).squeeze()
    )
    # 평균 유사도 계산
    average_similarity = df['similarity'].mean()
    threshold = 0  # 유사도 임계값 설정
    # 유사도 평균이 임계값을 초과하면 학교폭력 가능성을 판단
    print("뉴 학교 폭력 측정 지수 :", average_similarity)
    return average_similarity > threshold, average_similarity

bullying_keywords = {
    "학교폭력": 1,
    "괴롭힘": 1,
    "협박": 1,
    "위협": 1,
    "모욕": 1,
    "폭행": 1,
    "공격": 1,
    "가해": 1,
    "피해": 1,
    "목표": 1,
    "조롱": 1,
    "따돌림": 1,
    "포함제외": 1,
    "사회적편견": 1,
    "피해자": 1,
    "가해자": 1,
    "선생님괴롭힘": 1,
    "억지로쓰는글쓰기": 1,
    "협박하다": 1,
    "공격적": 1,
    "무력": 1,
    "폭력": 1,
    "억압": 1,
    "위협적": 1,
    "무서움": 1,
    "공포": 1
}

#GPT로 자세히 말하기 (지금 안써)
def generate_detailed_feedback(text, bullying_details):
    context = "Analyzing the provided text for indications of bullying based on keyword similarities: "
    for keyword, similarity in bullying_details.items():
        context += f"{keyword}: {similarity:.2f}, "
    context = context.rstrip(", ")  # 마지막 콤마 제거
    #prompt = f"{context}. 너의 역할은 심리상담 전문가야. 학교폭력을 당하고 있는 친구가 사용자일 확률이 높아. 이 분석을 기반으로 상황이 학교폭력과 관련되어 있는지에 대해 자세한 설명을 해줘. 학교폭력 가능성이 있다면, 주변에 도움을 받으라고 조언해줘. max_tokens 300을 넘기지마."
    prompt = f"{context}. 너의 역할은 심리상담 전문가야. 입력값 기반으로 상황이 학교폭력과 관련되어 있는지에 대해 전문성 있는 느낌 나게 분석 해줘. 최대 5문장으로 말해줘."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes texts for bullying."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=650
    )
    return response['choices'][0]['message']['content']



#대화 기록을 저장하기 위한 초기 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 대화 엔진 초기화 (가상의 함수, 실제 구현 필요)
def initialize_conversation():
    return None  # 여기에 실제 대화 엔진 초기화 로직 구현

if 'conversation' not in st.session_state:
    st.session_state.conversation = initialize_conversation()

# 사용자 입력 받기
query = st.chat_input("요즘 어떠세요. 하고 싶은말을 자유롭게 해보세요.")
if query:
    # 사용자 입력을 대화 기록에 저장
    st.session_state.history.append({"role": "user", "content": query})

    # 스피너를 사용하여 로딩 중 표시
    #with st.spinner("잠시만 기다려주세요..."):
    # 대화 엔진 또는 챗봇 모델로부터 응답 받기
    answer = generate_response(query)  # 가상의 응답 생성 함수

    # 응답 로직 분기
    print("--1--",answer )
    if answer['distance'] < 0.15:
        # 유사도가 낮으면 GPT 모델을 사용하여 응답 생성
        response = generate_gpt_response(query)
    else:
        # 유사도가 높은 경우, 기존 챗봇 응답 사용
        response = f"{answer['챗봇']}."

        # 챗봇 응답을 대화 기록에 저장
    st.session_state.history.append({"role": "assistant", "content": response})

# 대화 내용 순차적으로 출력
for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


#- 윗부분 메인페이지와 겹쳐서 막음(4/25)






# 사이드 바에 Streamlit UI
with st.sidebar:
    if st.sidebar.button("폭력 피해 탐지"):
        texts = [msg['content'] for msg in st.session_state.history if msg['role'] == 'user']
        bullying_results = [check_bullying(text, model) for text in texts]
        is_bullying_present = any(result[0] for result in bullying_results)  # 유사도 결과 검토
        average_indices = [result[1] for result in bullying_results]
        average_index = sum(average_indices) / len(average_indices) if average_indices else 0

        # 평균 유사도에 따른 5단계 결과 메시지 출력
        #print(average_index)
        average_percentage = round(average_index * 10000)
        print("--", average_percentage)
        st.write(f" * AI분석 통한 학교폭력 피해 가능성: {average_percentage}%")
        if 0 <= average_percentage <= 25:
           st.success("학교 폭력 피해 가능성 비교적 낮습니다.")
        elif 26 <= average_percentage <= 50:
           st.info("학교 폭력 피해 가능성이 보통이상으로 추가 상담이 필요합니다.")
        elif 51 <= average_percentage <= 100:
           st.error("학교 폭력 피해 가능성이 비교적 높은편으로 보호자의 관리가 필요합니다.")

        print("------------------------")
        # generate_detailed_feedback 함수 호출
        detailed_feedback = generate_detailed_feedback(texts, bullying_keywords)
        print("상세 피드백:")
        print(detailed_feedback)
    