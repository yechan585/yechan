mport streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 및 탭 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 최상단 아이콘 및 타이틀
st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()
total_count = len(df_raw)

# 정제: 네 속성 중 값이 없거나 첫 주 관객이 0인 영화 제거
# 속성: first_scrn, total_audi, days_in_top10, first_week_audi
df_clean = df_raw.dropna(subset=['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']).copy()
df_clean = df_clean[df_clean['first_week_audi'] > 0]

# 속성 변환
# 1. 스크린 수 상용로그
df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
# 2. 누적 관객 상용로그
df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])
# 3. 10위권 일수 (그대로 사용)
df_clean['days_in_top10_feat'] = df_clean['days_in_top10']
# 4. 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20으로 클리핑)
df_clean['long_run_index'] = df_clean['total_audi'] / df_clean['first_week_audi']
df_clean['long_run_index'] = np.minimum(df_clean['long_run_index'], 20.0)

used_count = len(df_clean)

# 전체 편수와 묶은 편수를 화면에 한 줄로 작성
st.write(f"**전체 영화 편수:** {total_count}편 | **묶음에 사용된 영화 편수:** {used_count}편")

st.divider()

# 3. 군집화에 사용할 속성 선택
feature_map = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10_feat",
    "롱런 지수": "long_run_index"
}

st.subheader("1. 군집화 속성 선택")
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(feature_map.keys()),
    default=list(feature_map.keys())
)

if len(selected_feature_names) < 2:
    st.warning("군집화를 진행하려면 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

selected_features = [feature_map[name] for name in selected_feature_names]

# 4. 표준화 및 K-Means 군집화
X = df_clean[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
initial_clusters = kmeans.fit_predict(X_scaled)
df_clean['raw_cluster'] = initial_clusters

# 누적 관객(total_audi) 평균이 큰 묶음부터 ㉮, ㉯, ㉰ 부여
cluster_avg_audi = df_clean.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
label_mapping = {raw_id: label for raw_id, label in zip(cluster_avg_audi.index, ['㉮', '㉯', '㉰'])}
df_clean['cluster'] = df_clean['raw_cluster'].map(label_mapping)

# 정렬을 위해 범주형 순서 지정
cluster_order = ['㉮', '㉯', '㉰']
df_clean['cluster'] = pd.Categorical(df_clean['cluster'], categories=cluster_order, ordered=True)

st.divider()

# 5. 시각화 섹션
st.subheader("2. 시각화")

col1, col2 = st.columns(2)

# 2차원 산점도
with col1:
    st.markdown("#### 2차원 산점도")
    x_axis_2d = st.selectbox("가로축 (X축) 속성:", options=list(feature_map.keys()), index=0, key="2d_x")
    y_axis_2d = st.selectbox("세로축 (Y축) 속성:", options=list(feature_map.keys()), index=1, key="2d_y")
    
    fig_2d = px.scatter(
        df_clean,
        x=feature_map[x_axis_2d],
        y=feature_map[y_axis_2d],
        color='cluster',
        category_orders={'cluster': cluster_order},
        hover_name='movieNm',
        labels={feature_map[x_axis_2d]: x_axis_2d, feature_map[y_axis_2d]: y_axis_2d, 'cluster': '묶음'},
        title=f"2D 산점도 ({x_axis_2d} vs {y_axis_2d})"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
with col2:
    st.markdown("#### 3차원 산점도")
    if len(selected_feature_names) < 3:
        st.info("💡 선택한 속성이 3개 미만입니다. 3차원 산점도를 그리려면 위에서 속성을 3개 이상 선택해주세요.")
    else:
        x_axis_3d = st.selectbox("X축 속성:", options=selected_feature_names, index=0, key="3d_x")
        y_axis_3d = st.selectbox("Y축 속성:", options=selected_feature_names, index=1, key="3d_y")
        z_axis_3d = st.selectbox("Z축 속성:", options=selected_feature_names, index=2, key="3d_z")
        
        fig_3d = px.scatter_3d(
            df_clean,
            x=feature_map[x_axis_3d],
            y=feature_map[y_axis_3d],
            z=feature_map[z_axis_3d],
            color='cluster',
            category_orders={'cluster': cluster_order},
            hover_name='movieNm',
            labels={
                feature_map[x_axis_3d]: x_axis_3d,
                feature_map[y_axis_3d]: y_axis_3d,
                feature_map[z_axis_3d]: z_axis_3d,
                'cluster': '묶음'
            },
            title=f"3D 산점도 ({x_axis_3d} / {y_axis_3d} / {z_axis_3d})"
        )
        fig_3d.update_traces(marker=dict(size=3))
        st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 6. 묶음별 요약 통계 및 대표 영화
st.subheader("3. 묶음별 요약 통계 및 관객 수 상위 영화")

# 원래 단위 속성 평균 표
summary_df = df_clean.groupby('cluster', observed=False).agg(
    편수=('movieNm', 'count'),
    스크린수_평균=('first_scrn', 'mean'),
    누적관객_평균=('total_audi', 'mean'),
    top10일수_평균=('days_in_top10', 'mean'),
    롱런지수_평균=('long_run_index', 'mean')
).reset_index()

summary_df.columns = ['묶음', '편수', '스크린 수 (평균)', '누적 관객 (평균)', '10위권 일수 (평균)', '롱런 지수 (평균)']

# 보기 좋게 포맷팅
formatted_summary = summary_df.copy()
formatted_summary['스크린 수 (평균)'] = formatted_summary['스크린 수 (평균)'].map('{:,.1f}'.format)
formatted_summary['누적 관객 (평균)'] = formatted_summary['누적 관객 (평균)'].map('{:,.1f}'.format)
formatted_summary['10위권 일수 (평균)'] = formatted_summary['10위권 일수 (평균)'].map('{:,.1f}'.format)
formatted_summary['롱런 지수 (평균)'] = formatted_summary['롱런 지수 (평균)'].map('{:,.2f}'.format)

st.write("#### 📊 묶음별 요약 표 (원래 단위 평균)")
st.dataframe(formatted_summary, hide_index=True, use_container_width=True)

st.write("#### 🏆 묶음별 누적 관객 수 Top 5 영화")
cols = st.columns(3)

for idx, label in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"##### 묶음 **{label}**")
        top_movies = df_clean[df_clean['cluster'] == label].sort_values(by='total_audi', ascending=False).head(5)
        for rank, (_, row) in enumerate(top_movies.iterrows(), 1):
            st.write(f"{rank}. **{row['movieNm']}** ({row['total_audi']:,}명)")
