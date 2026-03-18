from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Normalizer


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Telecom Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CUSTOM THEME / STYLING
# -----------------------------
def apply_custom_theme():
    st.markdown(
        """
        <style>
        :root {
            --primary-teal: #0f9d9a;
            --primary-teal-dark: #0b7f7c;
            --primary-teal-light: #e6f7f7;
            --teal-accent: #14b8b5;
            --magenta-accent: #c2185b;
            --navy-text: #1f2937;
            --soft-gray: #f7f8fa;
            --card-border: #ececec;
            --white: #ffffff;
        }

        .stApp {
            background-color: var(--soft-gray);
            color: var(--navy-text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f9d9a 0%, #0b7f7c 55%, #075e5b 100%);
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        section[data-testid="stSidebar"] .stInfo {
            background-color: rgba(255,255,255,0.10) !important;
            border: 1px solid rgba(255,255,255,0.20) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--white);
            border: 1px solid var(--card-border);
            padding: 16px;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--primary-teal-dark) !important;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: var(--navy-text) !important;
        }

        h1, h2, h3 {
            color: var(--primary-teal-dark);
            font-weight: 800;
        }

        .dashboard-banner {
            background: linear-gradient(90deg, #0f9d9a 0%, #14b8b5 55%, #c2185b 100%);
            padding: 1rem 1.25rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        }

        .dashboard-banner h1 {
            color: white !important;
            margin: 0;
            padding: 0;
            font-size: 2rem;
        }

        .dashboard-banner p {
            margin: 0.4rem 0 0 0;
            font-size: 0.98rem;
            opacity: 0.98;
        }

        .section-card {
            background: white;
            border-radius: 16px;
            padding: 1rem 1rem 0.5rem 1rem;
            border: 1px solid var(--card-border);
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }

        .small-note {
            color: #5b6470;
            font-size: 0.92rem;
        }

        .stButton>button {
            background-color: var(--primary-teal);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.55rem 1rem;
        }

        .stButton>button:hover {
            background-color: var(--primary-teal-dark);
            color: white;
        }

        .stDownloadButton>button {
            background-color: var(--teal-accent);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 700;
        }

        .stSelectbox label, .stSlider label, .stCheckbox label {
            font-weight: 600 !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border: 1px solid #d9d9d9 !important;
        }

        .stDataFrame, .stTable {
            background: white;
            border-radius: 12px;
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, #0f9d9a, #14b8b5, #c2185b);
            margin: 1.2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_theme()


# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DEFAULT_DATA_FILE = DATA_DIR / "cleaned_data.csv"

# -----------------------------
# CONSTANTS
# -----------------------------
APP_PAIRS = {
    "social_media": ("social_media_dl_(bytes)", "social_media_ul_(bytes)"),
    "google": ("google_dl_(bytes)", "google_ul_(bytes)"),
    "email": ("email_dl_(bytes)", "email_ul_(bytes)"),
    "youtube": ("youtube_dl_(bytes)", "youtube_ul_(bytes)"),
    "netflix": ("netflix_dl_(bytes)", "netflix_ul_(bytes)"),
    "gaming": ("gaming_dl_(bytes)", "gaming_ul_(bytes)"),
    "other": ("other_dl_(bytes)", "other_ul_(bytes)"),
}

CHART_COLORS = ["#0f9d9a", "#14b8b5", "#c2185b", "#ff8c42", "#6c63ff", "#26a69a", "#8e24aa"]


# -----------------------------
# HELPERS
# -----------------------------
def standardize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    return col


def bytes_to_gb(series_or_value):
    return series_or_value / (1024 ** 3)


def bytes_to_mb(series_or_value):
    return series_or_value / (1024 ** 2)


def first_existing_column(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calculate_outliers(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper


def cap_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def apply_chart_theme(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1f2937"),
        title_font=dict(color="#0b7f7c", size=18),
        legend_title_font=dict(color="#0b7f7c"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]

    for dt_col in ["start", "end"]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    non_numeric_cols = {"start", "end", "handset_manufacturer", "handset_type", "msisdn/number", "msisdn"}
    for col in df.columns:
        if col not in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass

    if "total_data" not in df.columns:
        total_components = []
        for dl_col, ul_col in APP_PAIRS.values():
            if dl_col in df.columns and ul_col in df.columns:
                total_components.append(df[dl_col].fillna(0) + df[ul_col].fillna(0))

        if total_components:
            df["total_data"] = sum(total_components)
        else:
            df["total_data"] = 0

    return df


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    try:
        lower_name = file_name.lower()

        if lower_name.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
            df = pd.read_excel(BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")

        return preprocess_dataframe(df)

    except Exception as e:
        raise ValueError(f"Error loading file '{file_name}': {e}") from e


@st.cache_data(show_spinner=False)
def load_default_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Default dataset not found: {path}. Make sure Data/cleaned_data.csv exists in your repo."
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Default dataset must be CSV or Excel.")

    return preprocess_dataframe(df)


def get_data():
    st.sidebar.subheader("📂 Data Input")

    use_uploaded_file = st.sidebar.checkbox(
        "Use uploaded dataset instead of default cleaned_data.csv",
        value=False,
        help="Unchecked = app auto-loads Data/cleaned_data.csv. Checked = upload your own file.",
    )

    if use_uploaded_file:
        uploaded_file = st.sidebar.file_uploader(
            "Upload telecom dataset",
            type=["csv", "xlsx", "xls"],
            help="Upload a telecom dataset once and use it across all dashboard pages.",
        )

        if uploaded_file is None:
            st.sidebar.warning("Please upload a file, or uncheck the box to use the default dataset.")
            return None, None

        try:
            df = load_data_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
            return df, uploaded_file.name
        except Exception as e:
            st.error(str(e))
            return None, None

    try:
        df = load_default_data(str(DEFAULT_DATA_FILE))
        return df, str(DEFAULT_DATA_FILE.relative_to(BASE_DIR))
    except Exception as e:
        st.error(str(e))
        return None, None


def show_expected_format():
    expected_cols = [
        "msisdn/number",
        "handset_manufacturer",
        "handset_type",
        "bearer_id",
        "dur._(ms)",
        "total_data",
        "social_media_dl_(bytes)",
        "social_media_ul_(bytes)",
        "google_dl_(bytes)",
        "google_ul_(bytes)",
        "email_dl_(bytes)",
        "email_ul_(bytes)",
        "youtube_dl_(bytes)",
        "youtube_ul_(bytes)",
        "netflix_dl_(bytes)",
        "netflix_ul_(bytes)",
        "gaming_dl_(bytes)",
        "gaming_ul_(bytes)",
        "other_dl_(bytes)",
        "other_ul_(bytes)",
    ]

    st.subheader("📋 Expected Data Format")
    st.write("Your CSV should ideally contain columns like these:")
    col1, col2 = st.columns(2)
    half = len(expected_cols) // 2
    with col1:
        for c in expected_cols[:half]:
            st.write(f"• {c}")
    with col2:
        for c in expected_cols[half:]:
            st.write(f"• {c}")


def show_banner(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="dashboard-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# PAGE 1: USER OVERVIEW
# -----------------------------
def user_overview_analysis(df: pd.DataFrame | None):
    show_banner(
        "Telecom Analytics Dashboard",
        "Product-oriented telecom overview: user base, device mix, data quality, behavior, and traffic patterns.",
    )

    if df is None:
        st.info("👈 Load the default dataset or upload a CSV file from the sidebar to begin the analysis.")
        show_expected_format()
        return

    user_col = first_existing_column(df, ["msisdn/number", "msisdn"])
    duration_col = first_existing_column(df, ["dur._(ms)", "duration"])

    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with col3:
        unique_users = df[user_col].nunique() if user_col else 0
        st.metric("Unique Users", f"{unique_users:,}")
    with col4:
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * max(df.shape[1], 1))) * 100
        st.metric("Missing Data %", f"{missing_percent:.2f}%")

    st.markdown("---")

    st.subheader("🔍 Data Quality Analysis")
    col1, col2 = st.columns(2)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    with col1:
        st.write("**Missing Values by Column**")
        if len(missing_data) > 0:
            missing_df = pd.DataFrame(
                {
                    "Column": missing_data.index,
                    "Missing Count": missing_data.values,
                    "Missing %": (missing_data.values / len(df)) * 100,
                }
            )
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found.")

    with col2:
        if len(missing_data) > 0:
            fig_missing = px.bar(
                x=missing_data.values[:10],
                y=missing_data.index[:10],
                orientation="h",
                title="Top 10 Columns with Missing Values",
                labels={"x": "Missing Count", "y": "Columns"},
                color=missing_data.index[:10],
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_missing), use_container_width=True)

    st.markdown("---")

    st.subheader("📱 Handset Analysis")
    if "handset_manufacturer" in df.columns and "handset_type" in df.columns:
        col1, col2 = st.columns(2)

        top_manufacturers = df["handset_manufacturer"].fillna("Unknown").value_counts().head(10)
        top_handsets = df["handset_type"].fillna("Unknown").value_counts().head(10)

        with col1:
            fig_manufacturers = px.bar(
                x=top_manufacturers.values,
                y=top_manufacturers.index,
                orientation="h",
                title="Top 10 Handset Manufacturers",
                labels={"x": "Count", "y": "Manufacturer"},
                color=top_manufacturers.index,
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_manufacturers), use_container_width=True)

        with col2:
            fig_handsets = px.bar(
                x=top_handsets.values,
                y=top_handsets.index,
                orientation="h",
                title="Top 10 Handset Types",
                labels={"x": "Count", "y": "Handset Type"},
                color=top_handsets.index,
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_handsets), use_container_width=True)

        st.write("**Top Handsets by Manufacturer**")
        manufacturer_options = list(top_manufacturers.index.unique())
        selected_manufacturer = st.selectbox("Select Manufacturer", manufacturer_options, key="manufacturer_select")

        manufacturer_handsets = (
            df[df["handset_manufacturer"] == selected_manufacturer]["handset_type"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
        )

        if len(manufacturer_handsets) > 0:
            fig_manu_handsets = px.bar(
                x=manufacturer_handsets.values,
                y=manufacturer_handsets.index,
                orientation="h",
                title=f"Top Handsets by {selected_manufacturer}",
                labels={"x": "Count", "y": "Handset Type"},
                color=manufacturer_handsets.index,
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_manu_handsets), use_container_width=True)
    else:
        st.warning("Handset columns not found in the dataset.")

    st.markdown("---")

    st.subheader("👥 User Behavior Analysis")
    if user_col:
        col1, col2 = st.columns(2)

        with col1:
            if "bearer_id" in df.columns:
                sessions = df.groupby(user_col)["bearer_id"].count().reset_index()
                sessions.columns = ["User", "Sessions"]
                sessions = sessions.sort_values("Sessions", ascending=False)

                st.write("**Session Statistics**")
                st.dataframe(sessions["Sessions"].describe().to_frame().T, use_container_width=True)

                st.write("**Top 10 Users by Sessions**")
                st.dataframe(sessions.head(10), use_container_width=True)
            else:
                st.info("Session column 'bearer_id' not available.")

        with col2:
            if duration_col:
                durations = df.groupby(user_col)[duration_col].sum().reset_index()
                durations.columns = ["User", "Total_Duration_ms"]
                durations = durations.sort_values("Total_Duration_ms", ascending=False)

                st.write("**Duration Statistics**")
                st.dataframe(durations["Total_Duration_ms"].describe().to_frame().T, use_container_width=True)

                st.write("**Top 10 Users by Duration**")
                durations_display = durations.copy()
                durations_display["Duration_Hours"] = durations_display["Total_Duration_ms"] / (1000 * 60 * 60)
                st.dataframe(durations_display[["User", "Duration_Hours"]].head(10), use_container_width=True)
            else:
                st.info("Duration column not available.")
    else:
        st.warning("User identifier column not found.")

    st.markdown("---")

    st.subheader("📊 Data Usage Analysis")
    app_data = {}
    for app, (dl_col, ul_col) in APP_PAIRS.items():
        if dl_col in df.columns and ul_col in df.columns:
            app_data[app.replace("_", " ").title()] = (df[dl_col].fillna(0) + df[ul_col].fillna(0)).sum()

    if app_data:
        app_data_gb = {k: v / (1024 ** 3) for k, v in app_data.items()}
        app_df = pd.DataFrame(list(app_data_gb.items()), columns=["Application", "Data_GB"]).sort_values(
            "Data_GB", ascending=False
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_apps = px.bar(
                app_df,
                x="Data_GB",
                y="Application",
                orientation="h",
                title="Total Data Usage by Application (GB)",
                labels={"Data_GB": "Data Usage (GB)", "Application": "Application"},
                color="Application",
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_apps), use_container_width=True)

        with col2:
            fig_pie = px.pie(
                app_df,
                values="Data_GB",
                names="Application",
                title="Data Usage Distribution by Application",
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_pie), use_container_width=True)
    else:
        st.warning("Application usage columns were not found.")

    st.markdown("---")

    if st.checkbox("Show Raw Data Preview", key="raw_preview_overview"):
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)


# -----------------------------
# PAGE 2: USER ENGAGEMENT
# -----------------------------
def user_engagement_analysis(df: pd.DataFrame | None):
    show_banner(
        "Telecom Analytics Dashboard",
        "Behavioral engagement analysis for segmentation, usage intensity, and telecom customer value profiling.",
    )

    if df is None:
        st.info("👈 Load the default dataset or upload a CSV file from the sidebar to begin the engagement analysis.")
        show_expected_format()
        return

    user_col = first_existing_column(df, ["msisdn/number", "msisdn"])
    duration_col = first_existing_column(df, ["dur._(ms)", "duration"])

    if not user_col:
        st.error("User column not found. Expected something like 'msisdn/number' or 'msisdn'.")
        return

    df_renamed = df.copy()
    if user_col != "msisdn":
        df_renamed = df_renamed.rename(columns={user_col: "msisdn"})
    if duration_col and duration_col != "duration":
        df_renamed = df_renamed.rename(columns={duration_col: "duration"})

    if "duration" not in df_renamed.columns:
        df_renamed["duration"] = 0
    if "bearer_id" not in df_renamed.columns:
        df_renamed["bearer_id"] = 1
    if "total_data" not in df_renamed.columns:
        df_renamed["total_data"] = 0

    engagement_metrics = (
        df_renamed.groupby("msisdn", dropna=False)
        .agg(
            sessions_frequency=("bearer_id", "count"),
            duration=("duration", "sum"),
            total_traffic=("total_data", "sum"),
        )
        .fillna(0)
    )

    st.subheader("📈 Engagement Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", f"{len(engagement_metrics):,}")
    with col2:
        st.metric("Avg Sessions/User", f"{engagement_metrics['sessions_frequency'].mean():.2f}")
    with col3:
        st.metric("Avg Duration (min)", f"{engagement_metrics['duration'].mean() / (1000 * 60):.2f}")
    with col4:
        if engagement_metrics["total_traffic"].sum() > 0:
            st.metric("Avg Traffic (MB)", f"{bytes_to_mb(engagement_metrics['total_traffic'].mean()):.2f}")
        else:
            st.metric("Avg Traffic", "N/A")

    st.markdown("---")

    st.subheader("🏆 Top Performers")
    col1, col2, col3 = st.columns(3)

    with col1:
        top_sessions = engagement_metrics.sort_values("sessions_frequency", ascending=False).head(10)
        fig_sessions = px.bar(
            x=top_sessions["sessions_frequency"],
            y=[str(idx) for idx in top_sessions.index],
            orientation="h",
            title="Top Users by Sessions",
            labels={"x": "Session Count", "y": "User"},
            color=top_sessions["sessions_frequency"],
            color_continuous_scale=["#e6f7f7", "#0f9d9a"],
        )
        st.plotly_chart(apply_chart_theme(fig_sessions, height=320), use_container_width=True)
        st.dataframe(top_sessions[["sessions_frequency"]], use_container_width=True)

    with col2:
        top_duration = engagement_metrics.sort_values("duration", ascending=False).head(10).copy()
        top_duration["duration_hours"] = top_duration["duration"] / (1000 * 60 * 60)
        fig_duration = px.bar(
            x=top_duration["duration_hours"],
            y=[str(idx) for idx in top_duration.index],
            orientation="h",
            title="Top Users by Duration (Hours)",
            labels={"x": "Duration (Hours)", "y": "User"},
            color=top_duration["duration_hours"],
            color_continuous_scale=["#d8f3f2", "#0f9d9a"],
        )
        st.plotly_chart(apply_chart_theme(fig_duration, height=320), use_container_width=True)
        st.dataframe(top_duration[["duration_hours"]], use_container_width=True)

    with col3:
        if engagement_metrics["total_traffic"].sum() > 0:
            top_traffic = engagement_metrics.sort_values("total_traffic", ascending=False).head(10).copy()
            top_traffic["traffic_gb"] = bytes_to_gb(top_traffic["total_traffic"])
            fig_traffic = px.bar(
                x=top_traffic["traffic_gb"],
                y=[str(idx) for idx in top_traffic.index],
                orientation="h",
                title="Top Users by Traffic (GB)",
                labels={"x": "Traffic (GB)", "y": "User"},
                color=top_traffic["traffic_gb"],
                color_continuous_scale=["#f7d7e4", "#c2185b"],
            )
            st.plotly_chart(apply_chart_theme(fig_traffic, height=320), use_container_width=True)
            st.dataframe(top_traffic[["traffic_gb"]], use_container_width=True)
        else:
            st.info("Traffic data not available.")

    st.markdown("---")

    st.subheader("📊 Statistical Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Engagement Metrics Statistics**")
        st.dataframe(engagement_metrics.describe(), use_container_width=True)

    with col2:
        available_distribution_metrics = ["sessions_frequency", "duration"]
        if engagement_metrics["total_traffic"].sum() > 0:
            available_distribution_metrics.append("total_traffic")

        metric_option = st.selectbox(
            "Select metric for distribution:",
            available_distribution_metrics,
            key="distribution_metric_select",
        )

        fig_dist = px.histogram(
            engagement_metrics,
            x=metric_option,
            nbins=30,
            title=f"Distribution of {metric_option.replace('_', ' ').title()}",
            labels={"x": metric_option.replace("_", " ").title(), "y": "Frequency"},
            color_discrete_sequence=["#0f9d9a"],
        )
        st.plotly_chart(apply_chart_theme(fig_dist), use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 Outlier Detection")
    col1, col2 = st.columns(2)

    metrics_for_box = ["sessions_frequency", "duration"]
    if engagement_metrics["total_traffic"].sum() > 0:
        metrics_for_box.append("total_traffic")

    with col1:
        selected_metric = st.selectbox(
            "Select metric for outlier analysis:",
            metrics_for_box,
            key="outlier_metric_select",
        )
        fig_box = px.box(
            engagement_metrics,
            y=selected_metric,
            title=f"Outlier Detection - {selected_metric.replace('_', ' ').title()}",
            labels={"y": selected_metric.replace("_", " ").title()},
            color_discrete_sequence=["#c2185b"],
        )
        st.plotly_chart(apply_chart_theme(fig_box), use_container_width=True)

    with col2:
        outlier_stats = {}
        for metric in metrics_for_box:
            count, lower, upper = calculate_outliers(engagement_metrics[metric])
            outlier_stats[metric] = {
                "Outlier Count": count,
                "Lower Bound": f"{lower:.2f}",
                "Upper Bound": f"{upper:.2f}",
                "Outlier %": f"{(count / len(engagement_metrics) * 100):.2f}%",
            }

        outlier_df = pd.DataFrame(outlier_stats).T
        st.dataframe(outlier_df, use_container_width=True)

        if st.button("Clean Outliers", key="clean_outliers_btn"):
            cleaned_metrics = engagement_metrics.copy()
            for col in metrics_for_box:
                cleaned_metrics[col] = cap_outliers(cleaned_metrics[col])

            st.success("Outliers cleaned successfully.")
            st.dataframe(cleaned_metrics.describe(), use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 User Engagement Clustering")
    clustering_data = engagement_metrics.copy()

    for col in metrics_for_box:
        clustering_data[col] = cap_outliers(clustering_data[col])

    scaler = MinMaxScaler()
    normalizer = Normalizer()
    scaled_data = scaler.fit_transform(clustering_data[metrics_for_box])
    normalized_data = normalizer.fit_transform(scaled_data)

    col1, col2 = st.columns(2)

    with col1:
        show_elbow = st.checkbox("Show Elbow Method", key="show_elbow_checkbox")
        if show_elbow:
            inertias = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(normalized_data)
                inertias.append(kmeans.inertia_)

            fig_elbow = px.line(
                x=list(k_range),
                y=inertias,
                markers=True,
                title="Elbow Method for Optimal k",
                labels={"x": "Number of Clusters (k)", "y": "Inertia"},
            )
            fig_elbow.update_traces(line=dict(color="#0f9d9a"))
            st.plotly_chart(apply_chart_theme(fig_elbow), use_container_width=True)

    with col2:
        n_clusters = st.slider("Select number of clusters:", 2, 8, 3, key="cluster_slider")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(normalized_data)

        clustered_data = clustering_data.copy()
        clustered_data["cluster"] = clusters

        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig_cluster_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Cluster Distribution",
            color_discrete_sequence=CHART_COLORS,
        )
        st.plotly_chart(apply_chart_theme(fig_cluster_pie), use_container_width=True)

        agg_dict = {
            "sessions_frequency": ["mean", "count"],
            "duration": ["mean"],
        }
        if "total_traffic" in clustered_data.columns:
            agg_dict["total_traffic"] = ["mean"]

        cluster_stats = clustered_data.groupby("cluster").agg(agg_dict).round(2)
        st.write("**Cluster Statistics**")
        st.dataframe(cluster_stats, use_container_width=True)

    st.write("**Cluster Visualization**")
    y_axis = "total_traffic" if clustered_data["total_traffic"].sum() > 0 else "sessions_frequency"
    plot_df = clustered_data.reset_index()

    fig_scatter = px.scatter(
        plot_df,
        x="duration",
        y=y_axis,
        color=plot_df["cluster"].astype(str),
        title="User Engagement Clusters",
        labels={"duration": "Duration", y_axis: y_axis.replace("_", " ").title(), "color": "Cluster"},
        hover_data=["msisdn"] if "msisdn" in plot_df.columns else None,
        color_discrete_sequence=CHART_COLORS,
    )
    st.plotly_chart(apply_chart_theme(fig_scatter), use_container_width=True)

    st.markdown("---")

    st.subheader("📱 Application Usage Analysis")
    available_app_cols = []

    for app, (dl_col, ul_col) in APP_PAIRS.items():
        if app in df_renamed.columns:
            available_app_cols.append(app)
        elif dl_col in df_renamed.columns and ul_col in df_renamed.columns:
            df_renamed[app] = df_renamed[dl_col].fillna(0) + df_renamed[ul_col].fillna(0)
            available_app_cols.append(app)

    if available_app_cols:
        app_metrics = df_renamed.groupby("msisdn")[available_app_cols].sum()

        app_total_df = pd.DataFrame(
            [{"app": app, "total": app_metrics[app].sum()} for app in available_app_cols]
        ).sort_values("total", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig_apps = px.bar(
                app_total_df,
                x="total",
                y="app",
                orientation="h",
                title="Total Data Usage by Application",
                labels={"total": "Total Data Volume", "app": "Application"},
                color="app",
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_apps), use_container_width=True)

        with col2:
            fig_pie_apps = px.pie(
                app_total_df,
                values="total",
                names="app",
                title="Application Usage Distribution",
                color_discrete_sequence=CHART_COLORS,
            )
            st.plotly_chart(apply_chart_theme(fig_pie_apps), use_container_width=True)

        st.write("**Top Users by Application**")
        n_top_users = st.slider("Number of top users to show:", 5, 20, 10, key="top_users_slider")
        selected_app = st.selectbox("Select application:", available_app_cols, key="selected_app_box")

        top_users_app = app_metrics.sort_values(selected_app, ascending=False).head(n_top_users)
        fig_top_users = px.bar(
            x=top_users_app.index.astype(str),
            y=top_users_app[selected_app],
            title=f"Top {n_top_users} {selected_app.title()} Users",
            labels={"x": "User ID", "y": f"{selected_app.title()} Usage"},
            color=top_users_app[selected_app],
            color_continuous_scale=["#e6f7f7", "#0f9d9a"],
        )
        st.plotly_chart(apply_chart_theme(fig_top_users), use_container_width=True)
    else:
        st.warning("Application usage data not found in the dataset.")

    if st.checkbox("Show Engagement Data Preview", key="engagement_preview"):
        st.subheader("📋 Engagement Data Preview")
        st.dataframe(engagement_metrics.head(100), use_container_width=True)


# -----------------------------
# PAGE 3: EXPERIENCE
# -----------------------------
def experience_analysis(df: pd.DataFrame | None):
    show_banner(
        "Telecom Analytics Dashboard",
        "Placeholder for user experience analytics such as reliability, latency, friction, and service quality.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bounce Rate", "32%", "-3%")
    with col2:
        st.metric("Load Time", "2.1s", "-0.3s")
    with col3:
        st.metric("Error Rate", "0.8%", "-0.2%")

    st.info("🚧 This page is ready for future expansion into QoE, network latency, service failure, and reliability metrics.")


# -----------------------------
# PAGE 4: SATISFACTION
# -----------------------------
def satisfaction_analysis(df: pd.DataFrame | None):
    show_banner(
        "Telecom Analytics Dashboard",
        "Placeholder for customer satisfaction, loyalty, sentiment, and value perception analytics.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Customer Satisfaction", "4.2/5", "0.1")
    with col2:
        st.metric("NPS Score", "67", "5")

    st.info("🚧 This page is ready for future expansion into NPS, churn signals, satisfaction drivers, and retention storytelling.")


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.sidebar.title("📊 Telecom Analytics Dashboard")
    st.sidebar.markdown("---")

    df, data_source = get_data()

    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "User Overview Analysis",
            "User Engagement Analysis",
            "Experience Analysis",
            "Satisfaction Analysis",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard communicates a Product Data Scientist workflow: user overview, engagement profiling, "
        "behavioral segmentation, experience monitoring, and satisfaction analytics."
    )

    if df is not None:
        st.sidebar.success(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        st.sidebar.caption(f"Source: {data_source}")
    else:
        st.sidebar.warning("No dataset loaded yet.")

    if page == "User Overview Analysis":
        user_overview_analysis(df)
    elif page == "User Engagement Analysis":
        user_engagement_analysis(df)
    elif page == "Experience Analysis":
        experience_analysis(df)
    elif page == "Satisfaction Analysis":
        satisfaction_analysis(df)


if __name__ == "__main__":
    main()