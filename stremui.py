import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Industrial Workforce Analytics",
    layout="wide",
    page_icon="📊"
)

# ================= LOAD DATA (GITHUB SAFE) =================
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "clean_workers_data.csv"

    df = pd.read_csv(DATA_PATH)

    df["State_Name"] = df["State_Name"].str.strip().str.title()
    df["Industry_Category"] = df["Industry_Category"].astype(str).str.strip()

    df = df[
        df["Industry_Category"].notna() &
        (df["Industry_Category"] != "") &
        (df["Industry_Category"].str.lower() != "none") &
        (df["Industry_Category"].str.lower() != "nan")
    ]

    df["Industry_Category_Clean"] = df["Industry_Category"].str.lower()
    df["Total_Workers"] = df["Main_Total"] + df["Marginal_Total"]

    return df

df = load_data()

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "",
    ["Home", "EDA", "State Analysis", "Industry Analysis"],
    label_visibility="collapsed"
)

# ================= HOME =================
if menu == "Home":
    st.title("🏭 Industrial Workforce Analysis in India")

    st.subheader("Problem Statement")
    st.markdown("""
In India, industrial workforce classification plays a critical role in understanding
employment patterns and economic development. Existing classifications of main and
marginal workers (excluding cultivators and agricultural laborers) are often outdated
and insufficient for current policy and planning needs.
""")

    st.subheader("Proposed Solution")
    st.markdown("""
• Clean and preprocess workforce data for accuracy  
• Perform exploratory data analysis (EDA)  
• Apply NLP techniques to industry descriptions  
• Use unsupervised machine learning for structural grouping  
• Visualize insights for policy and workforce planning  
""")

    st.subheader("Workflow Overview")
    st.markdown("""
**Data → NLP → Machine Learning → Insights**

• Data Cleaning & Feature Engineering  
• Text Vectorization (TF-IDF)  
• Unsupervised Learning (Clustering)  
• Insightful Visual Dashboards  
""")

# ================= EDA =================
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Industry Categories Word Cloud")

    industries_text = " ".join(df["Industry_Category_Clean"].unique())
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(industries_text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# ================= STATE ANALYSIS =================
elif menu == "State Analysis":
    st.title("🏛️ State-wise Workforce Analysis")

    state_df = df.groupby("State_Name").agg(
        Total=("Total_Workers", "sum"),
        Male=("Main_Male", "sum"),
        Female=("Main_Female", "sum"),
        Urban=("Main_Urban_Total", "sum"),
        Rural=("Main_Rural_Total", "sum")
    ).reset_index()

    state_df["Urban_Ratio"] = state_df["Urban"] / state_df["Total"]

    st.plotly_chart(
        px.bar(state_df, x="State_Name", y="Total",
               title="Total Workforce by State"),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(state_df, x="State_Name", y=["Male", "Female"],
               barmode="group",
               title="Gender Distribution by State"),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(state_df, x="State_Name", y=["Urban", "Rural"],
               title="Urban vs Rural Workforce by State"),
        use_container_width=True
    )

    st.plotly_chart(
        px.line(state_df, x="State_Name", y="Urban_Ratio",
                markers=True,
                title="Urban Workforce Ratio by State"),
        use_container_width=True
    )

    st.plotly_chart(
        px.pie(state_df, values="Total", names="State_Name",
               title="State-wise Workforce Share"),
        use_container_width=True
    )

# ================= INDUSTRY ANALYSIS =================
elif menu == "Industry Analysis":
    st.title("🏭 Industry-wise Workforce Analysis")

    # -------- NLP (Internal Only) --------
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(df["Industry_Category_Clean"])
    KMeans(n_clusters=5, random_state=42).fit(X)

    # -------- Meaningful Industry Analysis --------
    ind_df = df.groupby("Industry_Category").agg(
        Total=("Total_Workers", "sum"),
        Male=("Main_Male", "sum"),
        Female=("Main_Female", "sum"),
        Urban=("Main_Urban_Total", "sum"),
        Rural=("Main_Rural_Total", "sum")
    ).reset_index()

    ind_df["Female_Ratio"] = ind_df["Female"] / ind_df["Total"]

    st.plotly_chart(
        px.bar(ind_df, x="Total", y="Industry_Category",
               orientation="h",
               title="Total Workforce by Industry"),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(ind_df, x="Industry_Category", y=["Male", "Female"],
               title="Gender Distribution by Industry"),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(ind_df, x="Industry_Category", y=["Urban", "Rural"],
               barmode="group",
               title="Urban vs Rural Workforce by Industry"),
        use_container_width=True
    )

    st.plotly_chart(
        px.pie(ind_df, values="Total", names="Industry_Category",
               hole=0.45,
               title="Industry-wise Workforce Share"),
        use_container_width=True
    )

    st.plotly_chart(
        px.line(ind_df, x="Industry_Category", y="Female_Ratio",
                markers=True,
                title="Female Workforce Participation by Industry"),
        use_container_width=True
    )
