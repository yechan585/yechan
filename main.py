import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 설정 (탭 제목 및 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

# 제목 및 헤더
st.title("🎬 영화 유형 나누기")
st.write("K-평균(K-Means) 군집화 알고리즘을 활용하여 한국 영화 데이터를 세 가지 유형으로 분류합니다.")

# 2. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_and_preprocess_data():
    # 데이터 불러오기
    df_raw = pd.read_csv(DATA_URL, encoding="utf-8")
    total_raw_count = len(df_raw)

    df = df_raw.copy()

    # 필수 컬럼 결측치 제거 및 첫 주 관객 0 이하 제거
    required_cols = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi",
        "movieNm",
    ]
    df = df.dropna(subset=required_cols)
    df = df[df["first_week_audi"] > 0]
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0)]

    # 파생 속성 생성
    # 1) 스크린 수 상용로그
    df["log_first_scrn"] = np.log10(df["first_scrn"])
    # 2) 누적 관객 상용로그
    df["log_total_audi"] = np.log10(df["total_audi"])
    # 3) 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20으로 클리핑)
    df["long_run_index"] = df["total_audi"] / df["first_week_audi"]
    df["long_run_index"] = df["long_run_index"].clip(upper=20.0)

    clean_count = len(df)
    return df, total_raw_count, clean_count


df, total_count, clean_count = load_and_preprocess_data()

# 편수 정보 표시 (한 줄로)
st.info(
    f"📊 전체 영화 편수: **{total_count}편** | 분석 대상 영화 편수: **{clean_count}편** (결측치 및 첫 주 관객 0 제외)"
)

# 속성 매핑 사전
FEATURE_MAP = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 원본 단위 속성 매핑 (통계 표 표시용)
ORIGINAL_FEATURE_MAP = {
    "스크린 수(로그)": "first_scrn",
    "누적 관객(로그)": "total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 3. 군집화에 사용할 속성 선택
st.subheader("1. 묶는 데 사용할 속성 선택")
selected_feature_labels = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(FEATURE_MAP.keys()),
    default=list(FEATURE_MAP.keys()),
)

if len(selected_feature_labels) < 2:
    st.error("⚠️ 최소 2개 이상의 속성을 선택해야 군집화를 진행할 수 있습니다.")
    st.stop()

selected_features = [FEATURE_MAP[label] for label in selected_feature_labels]

# 4. K-Means 군집화 수행
X = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 난수 고정 random_state=42
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df["raw_cluster"] = raw_clusters

# 누적 관객(total_audi) 평균을 기준으로 ㉮, ㉯, ㉰ 군집 라벨 매핑
cluster_audi_mean = (
    df.groupby("raw_cluster")["total_audi"].mean().sort_values(ascending=False)
)
sorted_clusters = cluster_audi_mean.index.tolist()

label_mapping = {
    sorted_clusters[0]: "㉮",
    sorted_clusters[1]: "㉯",
    sorted_clusters[2]: "㉰",
}

df["cluster"] = df["raw_cluster"].map(label_mapping)

# 5. 시각화
st.subheader("2. 군집 시각화")

col1, col2 = st.columns(2)

# 2차원 산점도
with col1:
    st.markdown("#### 2차원 산점도")
    x_2d_label = st.selectbox(
        "2차원 X축 속성",
        options=selected_feature_labels,
        index=0,
        key="2d_x",
    )
    y_2d_label = st.selectbox(
        "2차원 Y축 속성",
        options=[l for l in selected_feature_labels if l != x_2d_label]
        or selected_feature_labels,
        index=min(1, len(selected_feature_labels) - 1),
        key="2d_y",
    )

    fig_2d = px.scatter(
        df,
        x=FEATURE_MAP[x_2d_label],
        y=FEATURE_MAP[y_2d_label],
        color="cluster",
        hover_name="movieNm",
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        labels={
            FEATURE_MAP[x_2d_label]: x_2d_label,
            FEATURE_MAP[y_2d_label]: y_2d_label,
            "cluster": "유형",
        },
        title=f"{x_2d_label} vs {y_2d_label}",
    )
    fig_2d.update_traces(marker=dict(size=6))
    st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.markdown("#### 3차원 산점도")
if len(selected_feature_labels) < 3:
    st.warning("💡 3차원 산점도를 표시하려면 속성을 3개 이상 선택해주세요.")
else:
    col_3d_x, col_3d_y, col_3d_z = st.columns(3)
    with col_3d_x:
        x_3d_label = st.selectbox(
            "3차원 X축 속성",
            options=selected_feature_labels,
            index=0,
            key="3d_x",
        )
    with col_3d_y:
        y_3d_label = st.selectbox(
            "3차원 Y축 속성",
            options=selected_feature_labels,
            index=1 if len(selected_feature_labels) > 1 else 0,
            key="3d_y",
        )
    with col_3d_z:
        z_3d_label = st.selectbox(
            "3차원 Z축 속성",
            options=selected_feature_labels,
            index=2 if len(selected_feature_labels) > 2 else 0,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=FEATURE_MAP[x_3d_label],
        y=FEATURE_MAP[y_3d_label],
        z=FEATURE_MAP[z_3d_label],
        color="cluster",
        hover_name="movieNm",
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        labels={
            FEATURE_MAP[x_3d_label]: x_3d_label,
            FEATURE_MAP[y_3d_label]: y_3d_label,
            FEATURE_MAP[z_3d_label]: z_3d_label,
            "cluster": "유형",
        },
        title=f"3D: {x_3d_label} × {y_3d_label} × {z_3d_label}",
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# 6. 묶음별 통계 및 주요 영화
st.subheader("3. 묶음별 통계 및 대표 영화")

# 묶음별 편수 및 원본 단위 평균 표
summary_list = []
for cluster_symbol in ["㉮", "㉯", "㉰"]:
    c_df = df[df["cluster"] == cluster_symbol]
    count = len(c_df)
    row = {"유형": cluster_symbol, "영화 편수": f"{count}편"}

    for label in list(FEATURE_MAP.keys()):
        orig_col = ORIGINAL_FEATURE_MAP[label]
        mean_val = c_df[orig_col].mean()
        if orig_col in ["first_scrn", "total_audi"]:
            row[f"평균 {label.split('(')[0]}"] = f"{mean_val:,.1f}"
        else:
            row[f"평균 {label}"] = f"{mean_val:,.2f}"

    summary_list.append(row)

summary_df = pd.DataFrame(summary_list)
st.markdown("##### 📌 유형별 편수 및 원본 단위 평균")
st.dataframe(summary_df, use_container_width=True)

# 묶음별 누적 관객 TOP 5 영화
st.markdown("##### 🏆 유형별 누적 관객 TOP 5 영화")
top5_cols = st.columns(3)

for idx, cluster_symbol in enumerate(["㉮", "㉯", "㉰"]):
    with top5_cols[idx]:
        st.markdown(f"### 유형 **{cluster_symbol}**")
        c_df = df[df["cluster"] == cluster_symbol]
        top5 = c_df.sort_values(by="total_audi", ascending=False).head(5)

        for rank, (_, row) in enumerate(top5.iterrows(), start=1):
            st.write(
                f"**{rank}. {row['movieNm']}** ({row['total_audi']:,}명)"
            )
