import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- 安全导入 AI 库 ---
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ✅ 防止未定义
uploaded_file = None
yearly_sums = None  # 给 AI 用，避免未定义

# ✅ 通用安全除法（避免 ZeroDivisionError）
def safe_div(num, den, default=np.nan):
    try:
        if den is None or den == 0:
            return default
        return num / den
    except Exception:
        return default

def fmt_num(x, fmt=".0f", na="—"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return na
    try:
        return format(float(x), fmt)
    except Exception:
        return na

# --- 页面配置 ---
st.set_page_config(page_title="水文气候智能分析系统", page_icon="🌊", layout="wide")

# --- 初始化 Session ---
if 'ai_report' not in st.session_state: st.session_state.ai_report = ""
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- 侧边栏：设置 ---
with st.sidebar:
    st.title("⚙️ 设置")

    # --- Gemini Key ---
    if HAS_GENAI:
        try:
            secrets_key = st.secrets.get("GEMINI_API_KEY", "")
        except FileNotFoundError:
            secrets_key = ""

        if secrets_key:
            st.success("✅ API Key 已加载")
            if st.toggle("临时使用其他 Key"):
                api_key = st.text_input("手动输入新 Key", type="password")
            else:
                api_key = secrets_key
        else:
            api_key = st.text_input("Gemini API Key", type="password", help="输入 Key 以开启 AI 功能")
    else:
        api_key = ""

    st.divider()
    data_source = st.radio("数据来源", ["生成模拟数据 (演示)", "上传 CSV 文件"])

    if data_source == "上传 CSV 文件":
        uploaded_file = st.file_uploader("请上传含 'date' 和 'rainfall' 列的文件", type="csv")

    st.divider()

    # ✅ 暴雨阈值输入框（全系统统一）
    storm_threshold = st.number_input(
        "暴雨阈值 (mm/日)",
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        help="用于暴雨天数/概率统计。常用：30/50/80 mm。"
    )

    # ✅ 暴雨频率口径切换
    storm_metric_mode = st.radio(
        "暴雨频率口径",
        ["次/年（暴雨天数/年份数）", "暴雨概率（暴雨天数/总天数）"],
        index=1,
        help="次/年衡量每年暴雨天数强度；暴雨概率更接近频率（暴雨日占比）。"
    )

# --- 1. 数据加载与处理 ---
@st.cache_data
def get_data(source, uploaded_file):
    if source == "生成模拟数据 (演示)":
        dates = pd.date_range(start="2004-01-01", end="2023-12-31", freq='D')
        np.random.seed(42)
        rainfall = np.random.exponential(scale=3, size=len(dates))
        seasonality = np.array([1 + 2.5 * np.sin(np.pi * (m-2) / 6) if 4 < m < 10 else 0.2 for m in dates.month])
        mask = np.random.rand(len(dates)) > 0.75
        yearly_factor = np.ones(len(dates))
        for i, d in enumerate(dates):
            if d.year == 2010: yearly_factor[i] = 1.5
            if d.year == 2015: yearly_factor[i] = 0.6
        final_rain = rainfall * seasonality * mask * 5 * yearly_factor
        df = pd.DataFrame({"date": dates, "rainfall": final_rain})
        df['rainfall'] = df['rainfall'].clip(lower=0).round(1)
    else:
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                # ✅ 列检查
                if 'date' not in df.columns or 'rainfall' not in df.columns:
                    st.error("CSV 必须包含 'date' 和 'rainfall' 两列（区分大小写）")
                    return None

                df['date'] = pd.to_datetime(df['date'], errors="coerce")
                df['rainfall'] = pd.to_numeric(df['rainfall'], errors="coerce").fillna(0.0)

                # ✅ 去掉无法解析日期的行
                df = df.dropna(subset=['date']).copy()
            except Exception as e:
                st.error(f"文件读取失败: {e}")
                return None
        else:
            return None

    if df is None or df.empty:
        return None

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df

df = get_data(data_source, uploaded_file if data_source == "上传 CSV 文件" else None)

# --- 主界面 ---
st.title("🌊 水文气候智能分析系统")

if df is None:
    st.info("👈 请在左侧选择数据来源，或查看「数据指南」下载样表。")
else:
    # --- 计算基础指标 ---
    yearly_sums = df.groupby('year', as_index=False)['rainfall'].sum()
    avg_annual = yearly_sums['rainfall'].mean()
    max_day = df['rainfall'].max()

    # 顶部指标栏（全局概览）
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年均降雨量", f"{fmt_num(avg_annual, '.0f')} mm")
    c2.metric("历史极值 (单日)", f"{fmt_num(max_day, '.1f')} mm")
    c3.metric("总降雨天数", f"{int((df['rainfall'] > 0.1).sum())} 天")
    c4.metric("记录年份", f"{df['year'].nunique()} 年")

st.markdown("---")

# === 分页功能区 ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 基础概览", "📅 日历热力图", "⚖️ 时段对比", "📉 深度水文", "🌊 洪水演进模拟", "💬 AI 助手", "📚 数据指南"]
)

# 只有当 df 存在时才渲染前6个 Tab
if df is not None:

    # --- Tab 1: 基础概览 ---
    with tab1:
        # ✅ 新增：阈值暴雨统计接入 Tab1 概览指标
        st.subheader("概览指标（含阈值暴雨统计）")

        storm_by_year = df.assign(is_storm=df["rainfall"] >= storm_threshold).groupby("year").agg(
            storm_days=("is_storm", "sum"),
            total_days=("is_storm", "size"),
        ).reset_index()
        storm_by_year["storm_prob"] = safe_div(storm_by_year["storm_days"], storm_by_year["total_days"], default=0.0)

        avg_storm_days = storm_by_year["storm_days"].mean() if len(storm_by_year) else np.nan
        avg_storm_prob = storm_by_year["storm_prob"].mean() if len(storm_by_year) else np.nan

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("年均降雨量", f"{fmt_num(avg_annual, '.0f')} mm")
        m2.metric("历史极值(单日)", f"{fmt_num(max_day, '.1f')} mm")
        m3.metric("记录年份", f"{df['year'].nunique()} 年")
        m4.metric("总降雨天数", f"{int((df['rainfall'] > 0.1).sum())} 天")
        m5.metric(f"多年平均暴雨天数 (≥{storm_threshold:g}mm)", f"{fmt_num(avg_storm_days, '.1f')} 天/年")
        m6.metric(f"多年平均暴雨概率 (≥{storm_threshold:g}mm)", f"{fmt_num(avg_storm_prob, '.1%')}")

        st.markdown("")

        col_charts_1, col_charts_2 = st.columns([2, 1])

        with col_charts_1:
            st.subheader("历年降雨总量 & 趋势")
            yearly_sums_local = df.groupby('year')['rainfall'].sum().reset_index()
            yearly_sums_local['MA_5'] = yearly_sums_local['rainfall'].rolling(window=5).mean()

            bar = alt.Chart(yearly_sums_local).mark_bar(color='#3b82f6', opacity=0.8).encode(
                x=alt.X('year:O', title='年份'),
                y=alt.Y('rainfall:Q', title='总降雨量 (mm)'),
                tooltip=['year', 'rainfall']
            )
            line = alt.Chart(yearly_sums_local).mark_line(color='#f59e0b', strokeWidth=3).encode(
                x='year:O', y='MA_5:Q',
                tooltip=[alt.Tooltip('MA_5', title='5年平均线', format='.0f')]
            )
            st.altair_chart((bar + line).interactive(), use_container_width=True)

        with col_charts_2:
            st.subheader("月度模式")
            monthly_avg = df.groupby('month')['rainfall'].mean().reset_index()
            line_chart = alt.Chart(monthly_avg).mark_area(
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color='#10b981', offset=0),
                        alt.GradientStop(color='white', offset=1)
                    ],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('month:O', title='月份'),
                y=alt.Y('rainfall:Q', title='日均降雨 (mm)'),
                tooltip=['month', 'rainfall']
            ).properties(height=350)
            st.altair_chart(line_chart, use_container_width=True)

    # --- Tab 2: 日历热力图 ---
    with tab2:
        st.subheader("🗓️ 每日降雨微观视图")
        selected_year = st.selectbox("选择年份查看详情:", sorted(df['year'].unique(), reverse=True))
        year_data = df[df['year'] == selected_year]
        heatmap = alt.Chart(year_data).mark_rect().encode(
            x=alt.X('day:O', title='日期'),
            y=alt.Y('month:O', title='月份'),
            color=alt.Color('rainfall:Q', scale=alt.Scale(scheme='blues'), title='降雨量(mm)'),
            tooltip=['date', 'rainfall']
        ).properties(width='container', height=400)
        st.altair_chart(heatmap, use_container_width=True)

    # --- Tab 3: 时段对比 ---
    with tab3:
        st.subheader("⚖️ 气候变化检测 (前半段 vs 后半段)")

        years = sorted(df['year'].unique())
        if len(years) < 2:
            st.info("年份太少，无法进行时段对比。请至少提供 2 年数据。")
        else:
            mid_point = len(years) // 2
            period_1 = years[:mid_point]
            period_2 = years[mid_point:]

            if len(period_1) == 0 or len(period_2) == 0:
                st.warning("分段后出现空时段，无法对比。建议至少 4 年数据效果更好。")
            else:
                df_p1 = df[df['year'].isin(period_1)].copy()
                df_p2 = df[df['year'].isin(period_2)].copy()

                avg_p1 = df_p1.groupby('year')['rainfall'].sum().mean()
                avg_p2 = df_p2.groupby('year')['rainfall'].sum().mean()

                # ✅ 暴雨统计统一使用 storm_threshold
                num1 = (df_p1["rainfall"] >= storm_threshold).sum()
                num2 = (df_p2["rainfall"] >= storm_threshold).sum()

                if storm_metric_mode == "次/年（暴雨天数/年份数）":
                    den1 = len(period_1)   # 年份数
                    den2 = len(period_2)
                    storm_p1 = safe_div(num1, den1, default=0.0)
                    storm_p2 = safe_div(num2, den2, default=0.0)
                    storm_label = f"暴雨频率变化 (≥{storm_threshold:g}mm)"
                    storm_fmt = ".1f"
                    storm_suffix = " 次/年"
                else:
                    den1 = len(df_p1)      # 总天数
                    den2 = len(df_p2)
                    storm_p1 = safe_div(num1, den1, default=0.0)
                    storm_p2 = safe_div(num2, den2, default=0.0)
                    storm_label = f"暴雨概率变化 (≥{storm_threshold:g}mm)"
                    storm_fmt = ".1%"
                    storm_suffix = ""

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("早期平均降雨", f"{fmt_num(avg_p1, '.0f')} mm", f"{period_1[0]}-{period_1[-1]}")
                col_b.metric("近期平均降雨", f"{fmt_num(avg_p2, '.0f')} mm", f"{fmt_num((avg_p2 - avg_p1), '.0f')} mm", delta_color="inverse")
                col_c.metric(
                    storm_label,
                    f"{format(storm_p2, storm_fmt)}{storm_suffix}",
                    f"{format(storm_p2 - storm_p1, storm_fmt)}",
                    delta_color="inverse"
                )

    # --- Tab 4: 深度水文分析 ---
    with tab4:
        c_hyd_1, c_hyd_2 = st.columns(2)

        with c_hyd_1:
            st.subheader("⛈️ 旱涝异常监测 (距平)")

            yearly_sums_local = df.groupby('year')['rainfall'].sum().reset_index()
            avg_annual_local = yearly_sums_local['rainfall'].mean()

            # ✅ 修复：avg_annual 可能为 0
            if avg_annual_local == 0 or np.isnan(avg_annual_local):
                st.warning("年均降雨量为 0 或无效，无法计算距平指数。")
                yearly_sums_local['anomaly_pct'] = 0.0
            else:
                yearly_sums_local['anomaly_pct'] = (yearly_sums_local['rainfall'] - avg_annual_local) / avg_annual_local

            anomaly_chart = alt.Chart(yearly_sums_local).mark_bar().encode(
                x=alt.X('year:O', title='年份'),
                y=alt.Y('anomaly_pct:Q', title='距平指数', axis=alt.Axis(format='%')),
                color=alt.condition(alt.datum.anomaly_pct > 0, alt.value("#3b82f6"), alt.value("#ef4444")),
                tooltip=[alt.Tooltip('year'), alt.Tooltip('anomaly_pct', format='.1%')]
            ).properties(height=300)
            st.altair_chart(anomaly_chart, use_container_width=True)

        with c_hyd_2:
            st.subheader("🌊 暴雨重现期推算（年最大日雨量）")
            annual_max = df.groupby('year')['rainfall'].max().sort_values(ascending=False).reset_index()
            n = len(annual_max)

            if n < 2:
                st.info("年份太少，无法推算重现期。请至少提供 2 年数据。")
            else:
                annual_max['rank'] = range(1, n + 1)
                annual_max['prob'] = annual_max['rank'] / (n + 1)
                annual_max['return_period'] = 1 / annual_max['prob']

                rp_chart = alt.Chart(annual_max).mark_circle(size=60, color='#f59e0b').encode(
                    x=alt.X('return_period:Q', title='重现期 (年)', scale=alt.Scale(type='log')),
                    y=alt.Y('rainfall:Q', title='日最大降雨量 (mm)'),
                    tooltip=['year', 'rainfall', alt.Tooltip('return_period', format='.1f')]
                ).properties(height=300)

                trend = rp_chart.transform_regression('return_period', 'rainfall', method='log').mark_line(color='gray')
                st.altair_chart(rp_chart + trend, use_container_width=True)

            # ✅ 新增：阈值暴雨统计（与 Tab3 同一阈值）
            st.markdown(f"---\n**阈值暴雨统计（≥{storm_threshold:g} mm/日）**")

            storm_by_year = df.assign(is_storm=df["rainfall"] >= storm_threshold).groupby("year").agg(
                storm_days=("is_storm", "sum"),
                total_days=("is_storm", "size"),
            ).reset_index()

            storm_by_year["storm_prob"] = storm_by_year["storm_days"] / storm_by_year["total_days"]

            avg_storm_days = storm_by_year["storm_days"].mean()
            avg_storm_prob = storm_by_year["storm_prob"].mean()

            mm1, mm2 = st.columns(2)
            mm1.metric("多年平均暴雨天数", f"{avg_storm_days:.1f} 天/年")
            mm2.metric("多年平均暴雨概率", f"{avg_storm_prob:.1%}")

            storm_trend = alt.Chart(storm_by_year).mark_line(point=True).encode(
                x=alt.X("year:O", title="年份"),
                y=alt.Y("storm_days:Q", title=f"暴雨天数 (≥{storm_threshold:g}mm)"),
                tooltip=["year", "storm_days", alt.Tooltip("storm_prob", format=".1%")]
            ).properties(height=220)
            st.altair_chart(storm_trend, use_container_width=True)

    # --- Tab 5: 洪水演进模拟 ---
    with tab5:
        st.subheader("🌊 河道洪水演进模拟 (马斯金根法)")
        st.caption("基于上游流量数据，模拟洪水在河道中的传播、滞后与削峰过程。")

        col_sim_1, col_sim_2 = st.columns([1, 2])

        with col_sim_1:
            st.markdown("#### 1. 设定洪水场景")
            peak_flow = st.slider("上游洪峰流量 (m³/s)", 100, 5000, 1000)
            base_flow = st.slider("基础流量 (m³/s)", 10, 500, 50)
            flood_duration = st.slider("洪水持续时间 (小时)", 10, 100, 24)

            st.markdown("#### 2. 设定河道参数")
            K = st.slider("传播时间 K (小时)", 1.0, 20.0, 5.0)
            X = st.slider("调蓄系数 X", 0.0, 0.5, 0.2)

        with col_sim_2:
            time_steps = np.arange(0, 100, 1)
            peak_time = 20
            inflow = base_flow + (peak_flow - base_flow) * np.exp(-0.5 * ((time_steps - peak_time) / (flood_duration / 4))**2)

            dt = 1.0
            denom = K * (1 - X) + 0.5 * dt

            if denom == 0:
                st.error("参数组合导致 denom=0（无法计算马斯金根系数），请调整 K/X。")
            else:
                C0 = (-K * X + 0.5 * dt) / denom
                C1 = (K * X + 0.5 * dt) / denom
                C2 = (K * (1 - X) - 0.5 * dt) / denom

                outflow = np.zeros(len(time_steps))
                outflow[0] = inflow[0]

                for t in range(1, len(time_steps)):
                    outflow[t] = C0 * inflow[t] + C1 * inflow[t-1] + C2 * outflow[t-1]
                    if outflow[t] < base_flow:
                        outflow[t] = base_flow

                sim_df = pd.DataFrame({'Time': time_steps, 'Inflow': inflow, 'Outflow': outflow}).melt(
                    'Time', var_name='Type', value_name='Flow'
                )

                hydrograph = alt.Chart(sim_df).mark_line(strokeWidth=3).encode(
                    x='Time', y='Flow',
                    color=alt.Color('Type', scale=alt.Scale(range=['#3b82f6', '#f59e0b'])),
                    tooltip=['Time', 'Type', 'Flow']
                ).properties(height=400, title="洪水演进过程线")
                st.altair_chart(hydrograph, use_container_width=True)

    # --- Tab 6: AI 数据对话 ---
    with tab6:
        st.subheader("💬 AI 水文数据助手")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("向数据提问..."):
            if not api_key:
                st.error("请先配置 API Key")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("AI 思考中..."):
                        try:
                            yearly_txt = yearly_sums.to_string(index=False) if yearly_sums is not None else ""
                            data_context = f"历年降雨(Year:mm):\n{yearly_txt}"
                            full_prompt = (
                                "你是一个水文数据助手。请基于给定数据，用简洁、可核验的方式回答。\n"
                                f"{data_context}\n"
                                f"暴雨阈值：≥{storm_threshold:g} mm/日\n"
                                f"问题: {prompt}"
                            )
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
                            res = model.generate_content(full_prompt)
                            st.markdown(res.text)
                            st.session_state.chat_history.append({"role": "assistant", "content": res.text})
                        except Exception as e:
                            st.error(f"Error: {e}")

# --- Tab 7: 数据指南 (无论是否有数据都显示) ---
with tab7:
    st.subheader("📚 数据准备与上传指南")

    st.markdown("""
    ### 1. 文件格式标准
    * **文件类型**：`.csv` (逗号分隔值文件)
    * **编码格式**：推荐 `UTF-8`

    ### 2. 数据列要求
    你的表格**必须**包含以下两列表头（区分大小写）：

    | 列名 (Header) | 数据类型 | 说明 | 示例 |
    | :--- | :--- | :--- | :--- |
    | **date** | 日期 | 格式 `YYYY-MM-DD` | `2004-01-01` |
    | **rainfall** | 数字 | 降雨量 (mm) | `25.4` |

    ### 3. 数据质量建议
    * **缺失值**：某天没下雨请填 `0`，不要留空。
    * **时间跨度**：建议至少 2 年的数据（对比分析更稳定），洪水模拟建议 10 年以上。
    """)

    st.divider()

    st.subheader("📥 下载标准样表")

    sample_data = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "rainfall": [0, 5.2, 12.8, 0, 0, 45.5, 2.1, 0, 0, 8.4]
    })

    csv = sample_data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="点击下载 CSV 样表 (template.csv)",
        data=csv,
        file_name="rainfall_template.csv",
        mime="text/csv",
    )
