import os
import json
import hashlib
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from numpy import dot, linalg
from openai import OpenAI

# ==================== 환경 변수 및 OpenAI 초기화 ====================
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EXCEL_PATH = os.getenv("EXCEL_PATH", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==================== 전역 설정 ====================
TARGET_SHEETS = ["병동분만실", "불편사항 대처"]   # 불편사항 대처는 전체에서만 노출됨

# ==================== 임베딩 캐시 ====================
def _emb_cache_path() -> str:
    return "/tmp/_emb_cache.json"

def _load_emb_cache() -> Dict[str, List[float]]:
    path = _emb_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_emb_cache(cache: Dict[str, List[float]]) -> None:
    try:
        with open(_emb_cache_path(), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(input=text, model="text-embedding-3-large")
    return resp.data[0].embedding

def safe_get_embedding(text: str, max_len: int = 2000) -> List[float]:
    if not text:
        text = " "
    if len(text) > max_len:
        text = text[:max_len]
    cache = _load_emb_cache()
    key = _hash_text(text)
    if key in cache:
        return cache[key]
    emb = get_embedding(text)
    cache[key] = emb
    _save_emb_cache(cache)
    return emb

# ==================== 코사인 유사도 ====================
def cos_sim(A: np.ndarray, B: np.ndarray) -> float:
    denom = (linalg.norm(A) * linalg.norm(B))
    if denom == 0:
        return 0.0
    return float(dot(A, B) / denom)

# ==================== 엑셀 데이터 로드 ====================
def _pick(row: pd.Series, candidates: List[str], default: str = "") -> str:
    for name in candidates:
        if name in row.index:
            val = row[name]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return default

@st.cache_data(show_spinner=True)
def load_quiz_data() -> List[Dict[str, Any]]:
    REAL_EXCEL = "nursing_data.xlsx"
    xls = pd.ExcelFile(REAL_EXCEL, engine="openpyxl")
    all_problems: List[Dict[str, Any]] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(REAL_EXCEL, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        sheet_normalized = "불편사항 대처" if sheet in ["병동별 고객 응대", "불편사항 대처"] else sheet

        for idx, row in df.iterrows():
            pid = f"{sheet_normalized}_{idx}"
            situation = _pick(row, ["상황", "상황 설명", "상황내용"], default="")
            question = _pick(row, ["질문", "질의", "문제"], default="")
            standard_answer = _pick(row, ["모범답안", "모범답변", "표준답변"], default="")
            all_problems.append({
                "id": pid,
                "sheet": sheet_normalized,
                "situation": situation,
                "question": question,
                "standard_answer": standard_answer,
                "embedding": None,
            })
    return all_problems

def _ensure_problem_embedding(problem: Dict[str, Any]) -> List[float]:
    if problem.get("embedding") is None:
        problem["embedding"] = safe_get_embedding(problem["standard_answer"])
    return problem["embedding"]

# ==================== GPT 평가 프롬프트 ====================
def create_evaluation_prompt(user_answer: str, problem: Dict[str, Any], similarity: float) -> str:
    return f"""
너는 간호 교육 평가위원이다. 아래 학생 답변을 간단히 평가해라.

[출력 형식]
피드백: 학생 답변에 대한 짧은 총평 (2~3문장)

장점:
- 1~2개 bullet point

단점:
- 1~2개 bullet point

점수:
- 0~100점 사이 정수 1개

개선 답변:
- 학생 답변을 간단히 보완한 예시 (짧게)

[상황]
{problem['situation']}

[질문]
{problem['question']}

[표준답변]
{problem['standard_answer']}

[학생 답변]
{user_answer}

[유사도 점수]
{similarity:.2f}
"""

def generate_evaluation(prompt: str) -> str:
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 간호 교육 전문가이며 평가위원이다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return result.choices[0].message.content
    except Exception as e:
        return "채점 오류가 발생했습니다."

# ==================== Streamlit UI ====================
st.set_page_config(page_title="간호사 교육 챗봇", page_icon="🩺", layout="centered")
st.title("🩺 간호사 교육 챗봇")

# 데이터 로드
try:
    all_problems = load_quiz_data()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# 세션 상태
if "category" not in st.session_state:
    st.session_state.category = "전체"
if "problem_index" not in st.session_state:
    st.session_state.problem_index = -1  # 시작 전
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = ""

# ---------------- 카테고리 선택 ----------------
allowed = ["전체", "병동분만실"]
st.subheader("카테고리 선택")
category = st.radio("문제를 풀 카테고리", options=allowed, index=0)
st.session_state.category = category

# ▶️ 시작하기 버튼 (카테고리 선택 밑)
if st.session_state.problem_index == -1:
    if st.button("▶️ 시작하기", use_container_width=True):
        st.session_state.problem_index = 0
        st.session_state.last_feedback = ""
        st.rerun()

# 문제 리스트 필터링
if category == "병동분만실":
    problems = [p for p in all_problems if p["sheet"] == "병동분만실"]
else:
    problems = [p for p in all_problems if p["sheet"] in TARGET_SHEETS]

# ---------------- 문제 표시 ----------------
st.divider()
st.subheader("문제")
if 0 <= st.session_state.problem_index < len(problems):
    p = problems[st.session_state.problem_index]
    st.markdown(f"**📍 부서:** {p['sheet']}")
    st.markdown(f"**📋 상황:** {p['situation'] or '-'}")
    st.markdown(f"**❓ 질문:** {p['question'] or '-'}")
elif st.session_state.problem_index >= len(problems):
    st.success("🎉 모든 문제를 다 풀었습니다. 수고하셨습니다!")

# ---------------- 답안 입력 ----------------
if 0 <= st.session_state.problem_index < len(problems):
    st.subheader("나의 답변")
    current_pid = problems[st.session_state.problem_index]["id"]
    user_answer = st.text_area(
        "여기에 답변을 입력하세요",
        height=160,
        key=f"user_answer_{current_pid}"
    )

    # 채점하기
    if st.button("✅ 채점하기", type="primary"):
        problem = problems[st.session_state.problem_index]
        if not user_answer.strip():
            st.warning("답변을 입력해 주세요.")
        else:
            try:
                user_emb = safe_get_embedding(user_answer)
                std_emb = _ensure_problem_embedding(problem)
                similarity = cos_sim(np.array(user_emb), np.array(std_emb))
                prompt = create_evaluation_prompt(user_answer, problem, similarity)
                feedback = generate_evaluation(prompt)
                st.session_state.last_feedback = feedback
                st.success("채점 완료!")
            except Exception as e:
                st.error(f"채점 오류: {e}")

    if st.session_state.last_feedback:
        st.subheader("📊 채점 결과")
        st.markdown(st.session_state.last_feedback)

    # 다음 문제 / 카테고리 변경
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ 다음 문제", use_container_width=True):
            st.session_state.problem_index += 1
            st.session_state.last_feedback = ""
            st.rerun()
    with col2:
        if st.button("🔄 카테고리 변경", use_container_width=True):
            st.session_state.category = "전체"
            st.session_state.problem_index = -1
            st.session_state.last_feedback = ""
            st.rerun()
