# -*- coding: utf-8 -*-
"""
duplicate_analyzer.py
아마존 PPC 키워드 중복 탐지 + 그루핑 가이드 모듈

입력: 아마존 광고 벌크파일(Sponsored Products Campaigns 시트) 또는 타겟팅 리포트 DataFrame
- 필요한 컬럼(자동 매핑 지원): 캠페인명, 광고그룹명, 키워드 텍스트, 매치타입, 상태,
  Impressions, Clicks, Spend, Sales, Orders

기존 streamlit 앱에 탭 하나 추가해서 render_duplicate_tab(df)만 호출하면 됨.
"""

import re
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# 1. 컬럼 자동 매핑 (벌크파일/타겟팅리포트/한국어 콘솔 모두 대응)
# ─────────────────────────────────────────────
COLUMN_ALIASES = {
    "campaign":  ["campaign name", "campaign name (informational only)", "campaign", "캠페인", "캠페인 이름"],
    "ad_group":  ["ad group name", "ad group name (informational only)", "ad group", "adgroup", "광고그룹", "광고 그룹 이름"],
    "keyword":   ["keyword text", "targeting", "keyword", "customer search term", "키워드", "타겟팅"],
    "match":     ["match type", "match", "매치 유형", "매치타입"],
    "state":     ["state", "status", "상태", "keyword state"],
    "impressions": ["impressions", "노출수", "노출"],
    "clicks":    ["clicks", "클릭수", "클릭"],
    "spend":     ["spend", "cost", "광고비", "지출"],
    "sales":     ["sales", "7 day total sales", "14 day total sales", "매출", "7일 총 매출"],
    "orders":    ["orders", "7 day total orders (#)", "주문수", "7일 총 주문 수(#)"],
}

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 표준 키로 변환. 못 찾은 지표 컬럼은 0으로 채움."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    for std, aliases in COLUMN_ALIASES.items():
        found = None
        for a in aliases:
            if a in lower_cols:
                found = lower_cols[a]
                break
        if found is not None:
            out[std] = df[found]
        elif std in ("impressions", "clicks", "spend", "sales", "orders"):
            out[std] = 0
        else:
            out[std] = ""
    # 숫자형 변환
    for c in ("impressions", "clicks", "spend", "sales", "orders"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


# ─────────────────────────────────────────────
# 2. 키워드 정규화
# ─────────────────────────────────────────────
def normalize_kw(kw: str) -> str:
    kw = str(kw).lower().strip()
    kw = re.sub(r"[\"'+\[\]]", "", kw)      # 벌크파일의 따옴표/대괄호/+ 제거
    kw = re.sub(r"\s+", " ", kw)
    return kw

def norm_match(m: str) -> str:
    m = str(m).lower().strip()
    if "exact" in m or "정확" in m:
        return "exact"
    if "phrase" in m or "구문" in m:
        return "phrase"
    if "broad" in m or "확장" in m:
        return "broad"
    return m

def is_enabled(s: str) -> bool:
    s = str(s).lower().strip()
    return s in ("enabled", "활성", "활성화됨", "running", "")  # 상태 컬럼 없으면 전부 포함


# ─────────────────────────────────────────────
# 3. 중복 탐지 (3단계)
# ─────────────────────────────────────────────
def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = map_columns(df_raw)
    df["kw_norm"] = df["keyword"].map(normalize_kw)
    df["match_norm"] = df["match"].map(norm_match)
    df = df[df["kw_norm"] != ""]
    df = df[df["state"].map(is_enabled)]
    df = df[df["match_norm"].isin(["exact", "phrase", "broad"])]
    # 같은 캠페인·광고그룹·키워드·매치타입은 합산(리포트 기간별 행 분리 대응)
    df = (df.groupby(["campaign", "ad_group", "kw_norm", "match_norm"], as_index=False)
            .agg({"impressions": "sum", "clicks": "sum", "spend": "sum",
                  "sales": "sum", "orders": "sum"}))
    df["acos"] = df.apply(lambda r: r["spend"] / r["sales"] if r["sales"] > 0 else None, axis=1)
    return df

def find_hard_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """레벨1: 동일 키워드 + 동일 매치타입이 2개 이상 캠페인/광고그룹에 활성화"""
    g = df.groupby(["kw_norm", "match_norm"])
    dup_keys = g.size()[g.size() > 1].index
    rows = []
    for (kw, mt) in dup_keys:
        sub = df[(df["kw_norm"] == kw) & (df["match_norm"] == mt)].copy()
        # 성과 기준 승자 선정: 주문수 ↓ 클릭수 ↓ ACOS ↑ 순
        sub = sub.sort_values(by=["orders", "clicks"], ascending=False)
        sub["권장조치"] = ["✅ 유지 (성과 우위)"] + [
            "⏸️ 일시중지 + 해당 캠페인에 네거티브 Exact 추가"
        ] * (len(sub) - 1)
        sub["중복그룹"] = f"{kw} [{mt}]"
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def find_cross_match_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """레벨2: 동일 키워드가 서로 다른 매치타입으로 활성화 (Exact 트래픽 잠식 위험)"""
    g = df.groupby("kw_norm")["match_norm"].nunique()
    multi = g[g > 1].index
    rows = []
    for kw in multi:
        sub = df[df["kw_norm"] == kw].copy()
        has_exact = "exact" in set(sub["match_norm"])
        def advice(r):
            if r["match_norm"] == "exact":
                return "✅ 유지 (수확용 Exact)"
            if has_exact:
                return f"➕ 이 캠페인에 네거티브 Exact [{kw}] 추가 → Exact 캠페인으로 트래픽 격리"
            return "⚠️ Exact 캠페인 없음 → 성과 좋으면 Exact로 졸업(승격) 권장"
        sub["권장조치"] = sub.apply(advice, axis=1)
        sub["중복그룹"] = f"{kw} [cross-match]"
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def find_containment_overlap(df: pd.DataFrame, min_impressions: int = 100) -> pd.DataFrame:
    """레벨3: Broad/Phrase 키워드가 다른 키워드를 토큰 단위로 포함 → 잠재적 트래픽 겹침
    예: broad 'hair treatment' ⊇ exact 'no wash hair treatment'
    """
    df = df[df["impressions"] >= min_impressions]
    broads = df[df["match_norm"].isin(["broad", "phrase"])]
    rows = []
    kw_tokens = {kw: set(kw.split()) for kw in df["kw_norm"].unique()}
    for _, b in broads.iterrows():
        b_tok = kw_tokens[b["kw_norm"]]
        for kw, toks in kw_tokens.items():
            if kw == b["kw_norm"]:
                continue
            if b_tok and b_tok.issubset(toks):  # broad 토큰이 다른 키워드에 모두 포함
                targets = df[df["kw_norm"] == kw]
                for _, t in targets.iterrows():
                    if (t["campaign"], t["ad_group"]) == (b["campaign"], b["ad_group"]):
                        continue
                    rows.append({
                        "넓은 키워드": f'{b["kw_norm"]} [{b["match_norm"]}]',
                        "위치(넓은)": f'{b["campaign"]} > {b["ad_group"]}',
                        "겹치는 키워드": f'{t["kw_norm"]} [{t["match_norm"]}]',
                        "위치(겹침)": f'{t["campaign"]} > {t["ad_group"]}',
                        "권장조치": f'넓은 키워드 캠페인에 네거티브 추가: [{t["kw_norm"]}]',
                    })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


# ─────────────────────────────────────────────
# 4. 그루핑 가이드 (테마 기반 광고그룹 재구성 제안)
# ─────────────────────────────────────────────
STOPWORDS = {"for", "the", "a", "an", "of", "with", "and", "to", "in", "on"}

def suggest_grouping(df: pd.DataFrame, top_n_roots: int = 15) -> pd.DataFrame:
    """키워드를 루트 토큰(가장 빈출 명사) 기준으로 클러스터링해서
    '1 광고그룹 = 1 테마' 구조 제안."""
    kws = df[["kw_norm", "match_norm", "campaign", "ad_group",
              "clicks", "orders", "spend", "sales"]].drop_duplicates("kw_norm" )
    # 토큰 빈도
    freq = {}
    for kw in kws["kw_norm"]:
        for t in set(kw.split()):
            if t in STOPWORDS or len(t) <= 1:
                continue
            freq[t] = freq.get(t, 0) + 1
    roots = sorted(freq.items(), key=lambda x: -x[1])[:top_n_roots]
    root_list = [r for r, _ in roots if freq[r] >= 2]

    def assign_root(kw):
        toks = kw.split()
        for r in root_list:           # 빈도 높은 루트 우선
            if r in toks:
                return r
        return "(기타)"

    kws = kws.copy()
    kws["제안 테마(루트)"] = kws["kw_norm"].map(assign_root)
    summary = (kws.groupby("제안 테마(루트)")
                  .agg(키워드수=("kw_norm", "nunique"),
                       총클릭=("clicks", "sum"),
                       총주문=("orders", "sum"),
                       총지출=("spend", "sum"),
                       총매출=("sales", "sum"))
                  .sort_values("총지출", ascending=False)
                  .reset_index())
    return kws.sort_values(["제안 테마(루트)", "clicks"], ascending=[True, False]), summary

def audit_ad_group_themes(kws_with_root: pd.DataFrame) -> pd.DataFrame:
    """현재 광고그룹 안에 테마가 몇 개 섞여있는지 진단 (2개 이상이면 분리 권장)"""
    audit = (kws_with_root.groupby(["campaign", "ad_group"])
             .agg(테마수=("제안 테마(루트)", "nunique"),
                  키워드수=("kw_norm", "nunique"),
                  포함테마=("제안 테마(루트)", lambda s: ", ".join(sorted(set(s)))))
             .reset_index())
    audit["진단"] = audit["테마수"].map(
        lambda n: "✅ 단일 테마 — 양호" if n == 1 else f"⚠️ {n}개 테마 혼재 — 광고그룹 분리 권장")
    return audit.sort_values("테마수", ascending=False)


# ─────────────────────────────────────────────
# 5. Streamlit 렌더링 (기존 앱에서 이 함수만 호출)
# ─────────────────────────────────────────────
def render_duplicate_tab(df_raw: pd.DataFrame):
    st.header("🔍 키워드 중복 탐지 & 그루핑 가이드")
    df = prepare(df_raw)
    if df.empty:
        st.warning("키워드 행을 찾지 못했습니다. 벌크파일의 'Sponsored Products Campaigns' 시트 또는 타겟팅 리포트를 업로드해주세요.")
        return

    st.caption(f"분석 대상: 활성 키워드 {df['kw_norm'].nunique():,}개 / {df[['campaign','ad_group']].drop_duplicates().shape[0]:,}개 광고그룹")

    # ── 레벨1
    st.subheader("1️⃣ 완전 중복 (같은 키워드 + 같은 매치타입)")
    st.caption("같은 검색어 입찰에 내 캠페인끼리 경쟁 → 입찰 통제력 상실, 데이터 분산. 성과 우위 1개만 남기세요.")
    hard = find_hard_duplicates(df)
    if hard.empty:
        st.success("완전 중복 없음 👍")
    else:
        st.error(f"{hard['중복그룹'].nunique()}개 키워드가 중복 타겟 중")
        st.dataframe(hard[["중복그룹", "campaign", "ad_group", "impressions", "clicks",
                           "spend", "sales", "orders", "acos", "권장조치"]],
                     use_container_width=True)
        st.download_button("중복 목록 CSV 다운로드",
                           hard.to_csv(index=False).encode("utf-8-sig"),
                           "hard_duplicates.csv", "text/csv")

    # ── 레벨2
    st.subheader("2️⃣ 매치타입 교차 중복 (Exact 잠식 위험)")
    st.caption("같은 키워드가 Broad/Phrase에도 살아있으면 Exact 트래픽을 뺏어감 → 네거티브 격리 필요.")
    cross = find_cross_match_overlap(df)
    if cross.empty:
        st.success("교차 매치 중복 없음 👍")
    else:
        st.warning(f"{cross['kw_norm'].nunique()}개 키워드가 복수 매치타입으로 활성화")
        st.dataframe(cross[["중복그룹", "match_norm", "campaign", "ad_group",
                            "clicks", "spend", "sales", "orders", "권장조치"]],
                     use_container_width=True)

    # ── 레벨3
    st.subheader("3️⃣ 포함 관계 겹침 (Broad/Phrase 확장 잠식)")
    st.caption("넓은 키워드가 좁은 키워드의 검색어까지 흡수하는 구조. 넓은 쪽에 네거티브를 넣어 격리.")
    min_imp = st.slider("최소 노출수 필터", 0, 1000, 100, step=50)
    contain = find_containment_overlap(df, min_impressions=min_imp)
    if contain.empty:
        st.success("포함 관계 겹침 없음 (또는 필터 기준 미달) 👍")
    else:
        st.dataframe(contain, use_container_width=True)

    # ── 그루핑 가이드
    st.subheader("4️⃣ 그루핑 가이드 (1 광고그룹 = 1 테마)")
    kws_root, summary = suggest_grouping(df)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**제안 테마별 요약**")
        st.dataframe(summary, use_container_width=True)
    with c2:
        st.markdown("**현재 광고그룹 테마 혼재 진단**")
        st.dataframe(audit_ad_group_themes(kws_root), use_container_width=True)
    with st.expander("키워드별 테마 배정 상세 보기"):
        st.dataframe(kws_root, use_container_width=True)
    st.download_button("그루핑 제안 CSV 다운로드",
                       kws_root.to_csv(index=False).encode("utf-8-sig"),
                       "grouping_suggestion.csv", "text/csv")


# ─────────────────────────────────────────────
# 6. 단독 실행용: 파일 업로드 → 분석 (이 파일 하나로 완결)
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="키워드 중복 탐지", page_icon="🔍", layout="wide")
    st.title("🔍 아마존 PPC 키워드 중복 탐지 & 그루핑 가이드")
    st.caption("벌크파일(xlsx) 또는 타겟팅 리포트(xlsx/csv)를 올리면 자동 분석합니다.")

    up = st.file_uploader("파일 업로드", type=["xlsx", "csv"])
    if up is None:
        st.info("👆 아마존 광고 콘솔 → 벌크 작업 → 벌크파일 다운로드 후 그대로 올려주세요.")
        return

    try:
        if up.name.endswith(".csv"):
            df_raw = pd.read_csv(up)
        else:
            xls = pd.ExcelFile(up)
            # 벌크파일이면 'Sponsored Products Campaigns' 시트 자동 선택
            sheet = next((s for s in xls.sheet_names if "sponsored products" in s.lower()),
                         xls.sheet_names[0])
            if len(xls.sheet_names) > 1:
                sheet = st.selectbox("시트 선택", xls.sheet_names,
                                     index=xls.sheet_names.index(sheet))
            df_raw = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        st.error(f"파일을 읽지 못했습니다: {e}")
        return

    render_duplicate_tab(df_raw)


if __name__ == "__main__":
    main()
