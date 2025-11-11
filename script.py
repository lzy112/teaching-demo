# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# 设置页面
st.set_page_config(
    page_title="抽样分布教学演示",
    page_icon="📊",
    layout="wide"
)


# 创建总体数据
@st.cache_data
def create_population(n=10000):
    np.random.seed(42)
    gender = np.random.choice(['男', '女'], size=n, p=[0.5, 0.5])
    heights = np.zeros(n)

    for i in range(n):
        if gender[i] == '男':
            heights[i] = np.random.normal(175, 6)
        else:
            heights[i] = np.random.normal(162, 5)

    return pd.DataFrame({
        '志愿者ID': range(1, n + 1),
        '性别': gender,
        '身高_cm': np.round(heights, 1)
    })


# 标题
st.title("📊 抽样分布交互式教学演示")
st.markdown("---")

# 创建总体数据
volunteers_df = create_population()
heights = volunteers_df['身高_cm'].values
gender = volunteers_df['性别'].values
true_mean = np.mean(heights)
true_std = np.std(heights)

# 侧边栏控制面板
st.sidebar.header("控制面板")

sample_size = st.sidebar.slider(
    "样本量 (n)",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

n_samples = st.sidebar.slider(
    "抽样次数",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

bias_level = st.sidebar.slider(
    "抽样偏差",
    min_value=0.0,
    max_value=0.8,
    value=0.0,
    step=0.1,
    help="0表示无偏，值越大表示越偏向男性"
)


# 模拟抽样函数
def simulate_sampling(bias=0.0):
    np.random.seed(42)
    sample_means = []

    for i in range(n_samples):
        if bias > 0:
            male_indices = np.where(gender == '男')[0]
            female_indices = np.where(gender == '女')[0]

            n_male = int(sample_size * (0.5 + bias / 2))
            n_female = sample_size - n_male

            male_sample = np.random.choice(heights[male_indices], n_male, replace=False)
            female_sample = np.random.choice(heights[female_indices], n_female, replace=False)
            sample = np.concatenate([male_sample, female_sample])
        else:
            sample = np.random.choice(heights, sample_size, replace=False)

        sample_means.append(np.mean(sample))

    return sample_means


# 执行抽样
sample_means = simulate_sampling(bias_level)
sampling_mean = np.mean(sample_means)
sampling_std = np.std(sample_means)

# 显示总体信息
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("真实总体均值", f"{true_mean:.2f} cm")
with col2:
    st.metric("抽样分布均值", f"{sampling_mean:.2f} cm")
with col3:
    bias = sampling_mean - true_mean
    st.metric("估计偏差", f"{bias:+.2f} cm")

# 创建图表
tab1, tab2, tab3 = st.tabs(["总体分布", "抽样分布", "变异性分析"])

with tab1:
    # 总体分布图
    fig1 = px.histogram(
        volunteers_df,
        x='身高_cm',
        color='性别',
        nbins=30,
        barmode='overlay',
        title='全体志愿者身高分布',
        color_discrete_map={'男': 'lightblue', '女': 'lightpink'}
    )
    fig1.add_vline(x=true_mean, line_dash="dash", line_color="red")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # 抽样分布图
    fig2 = px.histogram(
        x=sample_means,
        nbins=100,
        title=f'抽样分布 (n={sample_size}, 偏差={bias_level})'
    )
    fig2.add_vline(x=true_mean, line_dash="dash", line_color="red")
    fig2.add_vline(x=sampling_mean, line_dash="dash", line_color="blue")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # 变异性分析
    sample_sizes = [30, 100, 200, 500]
    data = []

    for size in sample_sizes:
        if size <= sample_size:
            temp_means = simulate_sampling(0)
            for mean_val in temp_means:
                data.append({'样本量': f'n={size}', '平均身高': mean_val})

    if data:
        df_variability = pd.DataFrame(data)
        fig3 = px.box(df_variability, x='样本量', y='平均身高', title='不同样本量下的变异性')
        fig3.add_hline(y=true_mean, line_dash="dash", line_color="red")
        st.plotly_chart(fig3, use_container_width=True)

# 服装尺码推荐
st.markdown("---")
st.header("👕 服装尺码推荐方案")


def recommend_clothing(sample_mean, sample_std, total=10000):
    size_ranges = {
        'S': (sample_mean - 3 * sample_std, sample_mean - 1 * sample_std),
        'M': (sample_mean - 1 * sample_std, sample_mean + 0 * sample_std),
        'L': (sample_mean + 0 * sample_std, sample_mean + 1 * sample_std),
        'XL': (sample_mean + 1 * sample_std, sample_mean + 3 * sample_std)
    }

    proportions = {}
    for size, (lower, upper) in size_ranges.items():
        prop = (stats.norm.cdf(upper, sample_mean, sample_std) -
                stats.norm.cdf(lower, sample_mean, sample_std))
        proportions[size] = prop

    quantities = {size: int(prop * total) for size, prop in proportions.items()}
    return quantities


quantities = recommend_clothing(sampling_mean, true_std)

# 显示尺码推荐
cols = st.columns(4)
sizes = ['S', 'M', 'L', 'XL']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (size, color) in enumerate(zip(sizes, colors)):
    with cols[i]:
        st.metric(
            label=f"{size} 码",
            value=f"{quantities[size]} 件",
            delta=f"{quantities[size] / 100:.1f}%"
        )

# 运行说明
st.sidebar.markdown("---")
st.sidebar.info("""
**使用说明:**
- 调整样本量观察变异性变化
- 调整偏差滑块理解有偏估计
- 观察抽样分布如何逼近真实均值
""")