# streamlit_app_english.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import math

# Page configuration
st.set_page_config(
    page_title="Sampling Distribution Teaching Demo",
    page_icon="📊",
    layout="wide"
)


# Create population data
@st.cache_data
def create_population(n=10000):
    np.random.seed(42)
    gender = np.random.choice(['Male', 'Female'], size=n, p=[0.5, 0.5])
    heights = np.zeros(n)

    for i in range(n):
        if gender[i] == 'Male':
            heights[i] = np.random.normal(175, 6)
        else:
            heights[i] = np.random.normal(162, 5)

    return pd.DataFrame({
        'Volunteer_ID': range(1, n + 1),
        'Gender': gender,
        'Height_cm': np.round(heights, 1)
    })


# Title
st.title("📊 Sampling Distribution Interactive Teaching Demo")
st.markdown("---")

# Create population data
volunteers_df = create_population()
heights = volunteers_df['Height_cm'].values
gender = volunteers_df['Gender'].values
true_mean = np.mean(heights)
true_std = np.std(heights)
true_variance = np.var(heights)

# Sidebar control panel
st.sidebar.header("🎛️ Control Panel")

# Sampling parameters
st.sidebar.subheader("Sampling Parameters")
sample_size = st.sidebar.slider(
    "Sample Size (n)",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Size of each sample drawn from the population"
)

n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Number of times to repeat the sampling process"
)

bias_level = st.sidebar.slider(
    "Sampling Bias",
    min_value=0.0,
    max_value=0.8,
    value=0.0,
    step=0.1,
    help="0 = Unbiased, higher values = biased toward males"
)

# Clothing size parameters
st.sidebar.subheader("Clothing Size Parameters")
st.sidebar.markdown("**Standard Deviations from Mean:**")

s_lower = st.sidebar.slider("S: Lower bound (σ from mean)", -3.0, 0.0, -2.0, 0.1)
s_upper = st.sidebar.slider("S: Upper bound (σ from mean)", -2.0, 0.0, -0.5, 0.1)
m_lower = st.sidebar.slider("M: Lower bound (σ from mean)", -1.0, 1.0, -0.5, 0.1)
m_upper = st.sidebar.slider("M: Upper bound (σ from mean)", -0.5, 1.0, 0.5, 0.1)
l_lower = st.sidebar.slider("L: Lower bound (σ from mean)", 0.0, 2.0, 0.5, 0.1)
l_upper = st.sidebar.slider("L: Upper bound (σ from mean)", 0.5, 2.0, 1.5, 0.1)
xl_lower = st.sidebar.slider("XL: Lower bound (σ from mean)", 1.0, 3.0, 1.5, 0.1)
xl_upper = st.sidebar.slider("XL: Upper bound (σ from mean)", 1.5, 4.0, 3.0, 0.1)

# Display statistical formulas
st.sidebar.subheader("📐 Statistical Formulas")
st.sidebar.latex(r"\text{Population Mean: } \mu = \frac{1}{N}\sum_{i=1}^{N} x_i")
st.sidebar.latex(r"\text{Sample Mean: } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i")
st.sidebar.latex(r"\text{Standard Error: } SE = \frac{\sigma}{\sqrt{n}}")
st.sidebar.latex(r"\text{Z-score: } z = \frac{x - \mu}{\sigma}")

# Population statistics section
st.header("🎯 Population Parameters (Ground Truth)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Population Mean (μ)", f"{true_mean:.2f} cm")
with col2:
    st.metric("Population Std Dev (σ)", f"{true_std:.2f} cm")
with col3:
    st.metric("Population Variance (σ²)", f"{true_variance:.2f} cm²")
with col4:
    st.metric("Population Size (N)", "10,000")


# Sampling function
def simulate_sampling(bias=0.0):
    np.random.seed(42)
    sample_means = []
    sample_stds = []

    for i in range(n_samples):
        if bias > 0:
            male_indices = np.where(gender == 'Male')[0]
            female_indices = np.where(gender == 'Female')[0]

            n_male = int(sample_size * (0.5 + bias / 2))
            n_female = sample_size - n_male

            male_sample = np.random.choice(heights[male_indices], n_male, replace=False)
            female_sample = np.random.choice(heights[female_indices], n_female, replace=False)
            sample = np.concatenate([male_sample, female_sample])
        else:
            sample = np.random.choice(heights, sample_size, replace=False)

        sample_means.append(np.mean(sample))
        sample_stds.append(np.std(sample))

    return sample_means, sample_stds


# Perform sampling
sample_means, sample_stds = simulate_sampling(bias_level)
sampling_mean = np.mean(sample_means)
sampling_std = np.std(sample_means)
standard_error = true_std / math.sqrt(sample_size)

# Display sampling results
st.header("📈 Sampling Distribution Results")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sampling Distribution Mean", f"{sampling_mean:.2f} cm")
with col2:
    st.metric("Standard Error (SE)", f"{standard_error:.2f} cm")
with col3:
    bias = sampling_mean - true_mean
    st.metric("Estimation Bias", f"{bias:+.2f} cm")
with col4:
    st.metric("95% CI",
              f"({sampling_mean - 1.96 * standard_error:.1f}, {sampling_mean + 1.96 * standard_error:.1f}) cm")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(
    ["Population Distribution", "Sampling Distribution", "Variability Analysis", "Statistical Theory"])

with tab1:
    # Population distribution
    fig1 = px.histogram(
        volunteers_df,
        x='Height_cm',
        color='Gender',
        nbins=30,
        barmode='overlay',
        title='Population Height Distribution of Volunteers',
        color_discrete_map={'Male': 'lightblue', 'Female': 'lightpink'},
        labels={'Height_cm': 'Height (cm)', 'count': 'Frequency'}
    )
    fig1.add_vline(x=true_mean, line_dash="dash", line_color="red",
                   annotation_text=f"μ = {true_mean:.1f}cm")
    st.plotly_chart(fig1, use_container_width=True)

    # Show empirical rule
    st.subheader("📊 Empirical Rule (68-95-99.7 Rule)")
    col1, col2, col3 = st.columns(3)

    within_1sd = np.sum((heights >= true_mean - true_std) & (heights <= true_mean + true_std)) / len(heights) * 100
    within_2sd = np.sum((heights >= true_mean - 2 * true_std) & (heights <= true_mean + 2 * true_std)) / len(
        heights) * 100
    within_3sd = np.sum((heights >= true_mean - 3 * true_std) & (heights <= true_mean + 3 * true_std)) / len(
        heights) * 100

    col1.metric("Within μ ± 1σ", f"{within_1sd:.1f}%", "68.3% expected")
    col2.metric("Within μ ± 2σ", f"{within_2sd:.1f}%", "95.4% expected")
    col3.metric("Within μ ± 3σ", f"{within_3sd:.1f}%", "99.7% expected")

with tab2:
    # Sampling distribution
    fig2 = px.histogram(
        x=sample_means,
        nbins=30,
        title=f'Sampling Distribution of Sample Means (n={sample_size}, bias={bias_level})',
        labels={'x': 'Sample Mean (cm)', 'count': 'Density'}
    )
    fig2.add_vline(x=true_mean, line_dash="dash", line_color="red",
                   annotation_text=f"Population μ = {true_mean:.1f}cm")
    fig2.add_vline(x=sampling_mean, line_dash="dash", line_color="blue",
                   annotation_text=f"Sample x̄ = {sampling_mean:.1f}cm")

    # Add normal curve overlay
    x_norm = np.linspace(min(sample_means), max(sample_means), 100)
    y_norm = stats.norm.pdf(x_norm, sampling_mean, sampling_std)
    fig2.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines',
                              name='Normal Distribution', line=dict(color='red', width=2)))

    st.plotly_chart(fig2, use_container_width=True)

    # Central Limit Theorem explanation
    st.subheader("🎯 Central Limit Theorem (CLT)")
    st.latex(r"\text{As } n \to \infty, \bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)")
    st.markdown(r"""
    **Key Points:**
    - Regardless of population distribution shape, sampling distribution of means approaches normality
    - Mean of sampling distribution = Population mean (μ)
    - Variance of sampling distribution = σ²/n
    - Standard Error = σ/√n
    """)

with tab3:
    # Variability analysis
    sample_sizes = [30, 100, 200, 500]
    variability_data = []

    for size in sample_sizes:
        if size <= sample_size:
            temp_means, _ = simulate_sampling(0)
            for mean_val in temp_means:
                variability_data.append({'Sample Size': f'n={size}', 'Sample Mean': mean_val})

    if variability_data:
        df_variability = pd.DataFrame(variability_data)
        fig3 = px.box(df_variability, x='Sample Size', y='Sample Mean',
                      title='Effect of Sample Size on Variability (Standard Error)')
        fig3.add_hline(y=true_mean, line_dash="dash", line_color="red",
                       annotation_text=f"μ = {true_mean:.1f}cm")
        st.plotly_chart(fig3, use_container_width=True)

    # Show theoretical vs empirical standard error
    st.subheader("📐 Theoretical vs Empirical Standard Error")
    theoretical_se = true_std / math.sqrt(sample_size)
    empirical_se = np.std(sample_means)

    col1, col2 = st.columns(2)
    col1.metric("Theoretical SE (σ/√n)", f"{theoretical_se:.3f} cm")
    col2.metric("Empirical SE (from simulation)", f"{empirical_se:.3f} cm")

with tab4:
    # Statistical theory explanations
    st.subheader("📚 Statistical Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Parameters vs Statistics")
        st.latex(r"""
        \begin{aligned}
        \text{Parameter} &: \theta = \text{Population characteristic} \\
        \text{Statistic} &: \hat{\theta} = \text{Sample estimate} \\
        \text{Bias} &: E[\hat{\theta}] - \theta \\
        \text{Unbiased} &: E[\hat{\theta}] = \theta
        \end{aligned}
        """)

        st.markdown("### Properties of Good Estimators")
        st.markdown(r"""
        - **Unbiasedness**: \(E[\hat{\theta}] = \theta\)
        - **Consistency**: \(\hat{\theta} \xrightarrow{p} \theta\) as \(n \to \infty\)
        - **Efficiency**: Small variance
        - **Sufficiency**: Uses all sample information
        """)

    with col2:
        st.markdown("### Sampling Distribution Properties")
        st.latex(r"""
        \begin{aligned}
        E[\bar{X}] &= \mu \\
        Var(\bar{X}) &= \frac{\sigma^2}{n} \\
        SE(\bar{X}) &= \frac{\sigma}{\sqrt{n}} \\
        \bar{X} &\sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ (CLT)}
        \end{aligned}
        """)

        st.markdown("### Confidence Intervals")
        st.latex(r"""
        \begin{aligned}
        95\%\ CI &= \bar{x} \pm 1.96 \times \frac{\sigma}{\sqrt{n}} \\
        99\%\ CI &= \bar{x} \pm 2.58 \times \frac{\sigma}{\sqrt{n}}
        \end{aligned}
        """)

# Clothing size recommendation with customizable parameters
st.markdown("---")
st.header("👕 Clothing Size Recommendation System")


def recommend_clothing_custom(sample_mean, sample_std, total=10000):
    # Use the customizable standard deviation ranges
    size_ranges = {
        'S': (sample_mean + s_lower * sample_std, sample_mean + s_upper * sample_std),
        'M': (sample_mean + m_lower * sample_std, sample_mean + m_upper * sample_std),
        'L': (sample_mean + l_lower * sample_std, sample_mean + l_upper * sample_std),
        'XL': (sample_mean + xl_lower * sample_std, sample_mean + xl_upper * sample_std)
    }

    proportions = {}
    for size, (lower, upper) in size_ranges.items():
        prop = (stats.norm.cdf(upper, sample_mean, sample_std) -
                stats.norm.cdf(lower, sample_mean, sample_std))
        proportions[size] = prop

    quantities = {size: int(prop * total) for size, prop in proportions.items()}
    return quantities, size_ranges


quantities, size_ranges = recommend_clothing_custom(sampling_mean, true_std)

# Display size recommendations
st.subheader("📋 Recommended Size Distribution")

cols = st.columns(4)
sizes = ['S', 'M', 'L', 'XL']

# Fixed color definitions with proper hex codes
size_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (size, color) in enumerate(zip(sizes, size_colors)):
    with cols[i]:
        lower_cm = sampling_mean + [s_lower, m_lower, l_lower, xl_lower][i] * true_std
        upper_cm = sampling_mean + [s_upper, m_upper, l_upper, xl_upper][i] * true_std

        st.metric(
            label=f"{size} Size",
            value=f"{quantities[size]:,} units",
            delta=f"{quantities[size] / 100:.1f}%"
        )
        st.caption(f"Range: {lower_cm:.1f}cm - {upper_cm:.1f}cm")

# 修复颜色格式 - 只修改有问题的部分
# 在 "Show the distribution graphically" 部分

# Show the distribution graphically
st.subheader("📊 Size Distribution Visualization")

# Create a figure showing the normal distribution with size ranges
x = np.linspace(sampling_mean - 3.5 * true_std, sampling_mean + 3.5 * true_std, 1000)
y = stats.norm.pdf(x, sampling_mean, true_std)

fig4 = go.Figure()

# Add the normal curve
fig4.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Height Distribution',
                          line=dict(color='black', width=2)))

# Add shaded regions for each size - FIXED VERSION
for i, size in enumerate(sizes):
    lower = sampling_mean + [s_lower, m_lower, l_lower, xl_lower][i] * true_std
    upper = sampling_mean + [s_upper, m_upper, l_upper, xl_upper][i] * true_std

    mask = (x >= lower) & (x <= upper)

    # 修复颜色格式 - 使用rgba格式
    color_hex = size_colors[i]
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    rgba_color = f'rgba({r}, {g}, {b}, 0.5)'

    fig4.add_trace(go.Scatter(
        x=x[mask], y=y[mask],
        fill='tozeroy',
        fillcolor=rgba_color,  # 使用正确的rgba格式
        line=dict(width=0),
        name=f'{size} Size ({quantities[size] / 100:.1f}%)',
        showlegend=True
    ))

fig4.add_vline(x=sampling_mean, line_dash="dash", line_color="red",
               annotation_text=f"x̄ = {sampling_mean:.1f}cm")

fig4.update_layout(
    title="Clothing Size Distribution Based on Normal Distribution",
    xaxis_title="Height (cm)",
    yaxis_title="Probability Density",
    height=400
)

st.plotly_chart(fig4, use_container_width=True)
# Statistical summary
st.subheader("📈 Statistical Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Production", "10,000 units", "100%")
with col2:
    coverage = sum(quantities.values()) / 10000 * 100
    st.metric("Size Coverage", f"{coverage:.1f}%", "of population")
with col3:
    largest_size = max(quantities, key=quantities.get)
    st.metric("Most Common", largest_size, f"{quantities[largest_size] / 100:.1f}%")

# Run instructions
st.sidebar.markdown("---")
st.sidebar.info("""
**Usage Instructions:**
- Adjust sample size to observe variability changes
- Modify bias level to understand biased estimation  
- Change size boundaries to customize clothing distribution
- Observe how sampling distribution approximates population parameters
- Study the statistical formulas and their practical applications
""")

# Footer
st.markdown("---")
st.markdown("""
**Learning Objectives:**
- Understand sampling distributions and the Central Limit Theorem
- Learn about bias, variability, and standard error
- Apply statistical concepts to real-world problems
- Connect mathematical formulas with practical applications
""")