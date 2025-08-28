import os
import json
import hashlib
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from numpy import dot, linalg
from openai import OpenAI

# ==================== 환경 변수 및 OpenAI 초기화 ====================
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EXCEL_PATH = os.getenv("EXCEL_PATH", "")  # 비워두면 자동탐색 (현재는 미사용)
client = OpenAI(api_key=OPENAI_API_KEY)

# ==================== 전역 설정 ====================
TARGET_SHEETS = ["병동분만실", "병동별 고객 응대"]  # 허용 카테고리(시트)

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
    """여러 후보 컬럼명 중 존재하고 값이 있는 첫 항목을 문자열로 반환"""
    for name in candidates:
        if name in row.index:
            val = row[name]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return default

@st.cache_data(show_spinner=True)
def load_quiz_data() -> Tuple[Dict[str, pd.DataFrame], List[str], List[Dict[str, Any]]]:
    """
    텔레그램 버전과 동일한 컬럼 매핑 유지:
    - 상황: ["상황", "상황 설명", "상황내용"]
    - 질문: ["질문", "질의", "문제"]
    - 모범답안: ["모범답안", "모범답변", "표준답변"]
    """
    REAL_EXCEL = "nursing_data.xlsx"   # 같은 폴더에 있는 파일 고정
    xls = pd.ExcelFile(REAL_EXCEL, engine="openpyxl")
    data_dict: Dict[str, pd.DataFrame] = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(REAL_EXCEL, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]  # 컬럼 공백 제거
        data_dict[sheet] = df

    all_problems: List[Dict[str, Any]] = []
    for sheet, df in data_dict.items():
        for idx, row in df.iterrows():
            pid = f"{sheet}_{idx}"
            situation = _pick(row, ["상황", "상황 설명", "상황내용"], default="")
            question = _pick(row, ["질문", "질의", "문제"], default="")
            standard_answer = _pick(row, ["모범답안", "모범답변", "표준답변"], default="")
            all_problems.append({
                "id": pid,
                "sheet": sheet,
                "situation": situation,
                "question": question,
                "standard_answer": standard_answer,
                "embedding": None,
            })
    return data_dict, xls.sheet_names, all_problems

def _ensure_problem_embedding(problem: Dict[str, Any]) -> List[float]:
    if problem.get("embedding") is None:
        problem["embedding"] = safe_get_embedding(problem["standard_answer"])
    return problem["embedding"]

# ==================== 문제 추출 ====================
def _allowed_categories(sheet_names: List[str]) -> List[str]:
    filtered = [s for s in sheet_names if s in TARGET_SHEETS]
    return filtered if filtered else list(sheet_names)

def get_random_problem(all_problems: List[Dict[str, Any]], category: str | None = None) -> Dict[str, Any] | None:
    try:
        if category and category != "전체":
            filtered = [p for p in all_problems if p["sheet"] == category]
            return random.choice(filtered) if filtered else None
        return random.choice(all_problems) if all_problems else None
    except Exception:
        return None

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
        print(f"GPT 오류: {e}")
        return "죄송합니다. 채점 중 오류가 발생했습니다."

# ==================== Streamlit UI ====================
st.set_page_config(page_title="간호사 교육 챗봇 (엑셀 기반)", page_icon="🩺", layout="centered")
st.title("🩺 간호사 교육 챗봇")
st.caption("데이터 소스: Excel 파일 (DB 없이 동작)")

# 데이터 로드
try:
    data_dict, sheet_names, all_problems = load_quiz_data()
    st.caption(f"데이터 시트: {', '.join(sheet_names)}")
except Exception as e:
    st.error(f"데이터 로드 실패: {type(e).__name__}: {str(e)[:200]}")
    st.stop()

# 세션 상태 초기화
if "category" not in st.session_state:
    st.session_state.category = None
if "problem_id" not in st.session_state:
    st.session_state.problem_id = None
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = ""
if "last_problem" not in st.session_state:
    st.session_state.last_problem = None
if "prev_category" not in st.session_state:
    st.session_state.prev_category = None  # UI 전환용
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""      # ✨ 입력 유지/초기화용

# 카테고리 선택 (전체 + 허용 시트 두 개)
allowed = ["전체"] + _allowed_categories(list(sheet_names))
st.subheader("카테고리 선택")
# Streamlit 1.33+ segmented_control, 그 이하 버전이면 radio로 교체 가능
try:
    category = st.segmented_control("문제를 풀 카테고리", options=allowed, default=allowed[0], help="시트를 선택하세요.")
except Exception:
    category = st.radio("문제를 풀 카테고리", options=allowed, index=0, help="시트를 선택하세요.")
st.session_state.category = category

# 카테고리가 바뀌면 상태 초기화 (버튼 라벨 '새 문제'로 리셋 + 입력창 비움)
if st.session_state.prev_category != category:
    st.session_state.prev_category = category
    st.session_state.problem_id = None
    st.session_state.last_problem = None
    st.session_state.last_feedback = ""
    st.session_state.user_answer = ""      # ✨ 비우기

col_a, col_b = st.columns(2)

# ⬇️ 처음엔 '🎲 새 문제 받기', 이후엔 '➡️ 다음 문제'로 자동 전환
main_btn_label = "🎲 새 문제 받기" if st.session_state.last_problem is None else "➡️ 다음 문제"

with col_a:
    if st.button(main_btn_label, use_container_width=True):
        prob = get_random_problem(all_problems, st.session_state.category)
        if not prob:
            st.warning("⚠️ 해당 카테고리에서 문제를 찾을 수 없습니다.")
        else:
            st.session_state.problem_id = prob["id"]
            st.session_state.last_problem = prob
            st.session_state.last_feedback = ""
            st.session_state.user_answer = ""  # ✨ 다음 문제 받을 때 입력창 비우기

with col_b:
    if st.button("🔄 카테고리 초기화", use_container_width=True):
        st.session_state.category = allowed[0]
        st.session_state.prev_category = allowed[0]
        st.session_state.problem_id = None
        st.session_state.last_problem = None
        st.session_state.last_feedback = ""
        st.session_state.user_answer = ""      # ✨ 비우기

# 현재 문제 표시 (텔레그램 메시지 포맷 유지)
st.divider()
st.subheader("문제")
if st.session_state.last_problem:
    p = st.session_state.last_problem
    st.markdown(f"**📍 부서:** {p['sheet']}")
    st.markdown(f"**📋 상황:** {p['situation'] or '-'}")
    st.markdown(f"**❓ 질문:** {p['question'] or '-'}")
else:
    st.info("좌측 상단의 **‘🎲 새 문제 받기’**를 눌러 시작하세요.")

# 답안 입력
st.subheader("나의 답변")
user_answer = st.text_area(
    "여기에 답변을 입력하세요",
    height=160,
    placeholder="예) 불편을 드려 죄송합니다. 시설팀 점검을 요청하고, 예상 소요시간을 안내드리겠습니다...",
    key="user_answer"  # ✨ 세션 상태로 관리
)

# 채점 (유사도 → GPT 평가, 텔레그램과 동일 로직)
if st.button("✅ 채점하기", type="primary"):
    if not st.session_state.last_problem:
        st.warning("먼저 **‘🎲 새 문제 받기’**를 눌러 문제를 받아주세요.")
    elif not user_answer.strip():
        st.warning("답변을 입력해 주세요.")
    else:
        try:
            user_emb = safe_get_embedding(user_answer)
            std_emb = _ensure_problem_embedding(st.session_state.last_problem)
            similarity = cos_sim(np.array(user_emb), np.array(std_emb))
            prompt = create_evaluation_prompt(user_answer, st.session_state.last_problem, similarity)
            feedback = generate_evaluation(prompt)
            st.session_state.last_feedback = feedback
            st.success("채점 완료!")
        except Exception as e:
            st.error(f"채점 오류: {type(e).__name__}: {str(e)[:200]}")

# 결과 표시 (텔레그램의 채점 답장 포맷 그대로)
if st.session_state.last_feedback:
    st.subheader("📊 채점 결과")
    st.markdown(st.session_state.last_feedback)

# (옵션) 디버그
with st.expander("🔎 디버그(옵션)"):
    st.write("현재 카테고리:", st.session_state.category)
    st.write("현재 문제 ID:", st.session_state.problem_id)
    st.write("현재 입력값:", st.session_state.user_answer)  # 확인용
    if st.session_state.last_problem:
        st.json({
            "sheet": st.session_state.last_problem["sheet"],
            "situation": st.session_state.last_problem["situation"],
            "question": st.session_state.last_problem["question"],
        })

