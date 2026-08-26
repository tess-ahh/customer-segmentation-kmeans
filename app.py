import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Mall_Customers.csv")


df = load_data()

# --------------------------------------------------
# Automatic Cluster Naming
# --------------------------------------------------

def get_cluster_name(income, spending, income_median, spending_median):

    high_income = income >= income_median
    high_spending = spending >= spending_median

    if high_income and high_spending:
        return "Target Customers"

    elif high_income and not high_spending:
        return "Potential Customers"

    elif not high_income and high_spending:
        return "Impulse Buyers"

    else:
        return "Budget Customers"
# --------------------------------------------------
# Train K-Means Model
# --------------------------------------------------

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("👥 Customer Segmentation")
page = st.sidebar.radio(
    "📌Navigation",
    [
        "Home",
        "Dataset",
        "EDA",
        "K-Means Clustering",
        "Customer Segments"
    ]
)
st.sidebar.markdown("### 📌 Project Details")

st.sidebar.info(
    """
    **Algorithm:** K-Means Clustering

    **Features:**  
    Annual Income  
    Spending Score

    **Clusters:** 5

    **Silhouette Score:** 0.554
    """
)

# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "Home":

    st.title("👥 K-Means Clustering for Customer Segmentation")


    st.write(
        """
        This application uses the **K-Means Clustering** algorithm
        to segment mall customers based on their Annual Income
        and Spending Score.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Customers",
            len(df)
        )

    with col2:
        st.metric(
            "📊 Features",
            "2"
        )

    with col3:
        st.metric(
            "🎯 Optimal K",
            "5"
        )

    with col4:
        st.metric(
            "📈 Silhouette Score",
            "0.554"
        )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        The objective of this project is to identify meaningful
        customer groups based on purchasing behavior.

        These customer segments can help businesses develop
        targeted marketing strategies, personalized offers,
        and customer retention programs.
        """
    )
    # How It Works
    st.subheader("⚙️ How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Explore")
        st.write(
            "Analyze customer demographics and purchasing behavior."
        )

    with col2:
        st.markdown("### 2️⃣ Cluster")
        st.write(
            "Apply K-Means clustering to identify customer groups."
        )

    with col3:
        st.markdown("### 3️⃣ Understand")
        st.write(
            "Interpret each segment and derive business insights."
        )

    st.divider()
    st.subheader("📌 Features Used for Clustering")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Annual Income")
        st.write(
            "Customer annual income measured in thousands of dollars."
        )

    with col2:
        st.markdown("### 🛍️ Spending Score")
        st.write(
            "A score between 1 and 100 representing customer spending behavior."
        )

    st.divider()
    st.subheader("🛠️ Technologies Used")

    st.write(
        """
        **Python**  • **Pandas**  • **NumPy**  • **Scikit-learn**  •
        **Matplotlib**  • **Seaborn**  • **Streamlit**
        """
    )


# --------------------------------------------------
# DATASET
# --------------------------------------------------

elif page == "Dataset":

    st.title("📊 Dataset Explorer")

    st.subheader("Mall Customers Dataset")

    st.write(
        f"The dataset contains **{len(df)} customer records**."
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    st.dataframe(
        missing_values.rename("Missing Values"),
        use_container_width=True
    )


# --------------------------------------------------
# EDA
# --------------------------------------------------

elif page == "EDA":

    st.title("📈 Exploratory Data Analysis")

    # Gender Distribution

    st.subheader("Gender Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="Gender",
        ax=ax
    )

    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # Age Distribution

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        df["Age"],
        bins=15,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # Income Distribution

    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        df["Annual Income (k$)"],
        bins=15,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # Spending Score Distribution

    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        df["Spending Score (1-100)"],
        bins=15,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Spending Score")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    # Correlation Heatmap

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)


# --------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------

elif page == "K-Means Clustering":

    st.title("🤖 K-Means Customer Clustering")
    st.markdown(
        """
        Explore customer segments using the **K-Means clustering
        algorithm**. Adjust the number of clusters and observe
        how the segmentation changes.
        """
    )

    st.subheader("Elbow Method")

    wcss = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        model.fit(X)

        wcss.append(model.inertia_)

    fig, ax = plt.subplots()

    ax.plot(
        range(1, 11),
        wcss,
        marker="o"
    )

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")

    st.pyplot(fig)
    st.info(
        "The Elbow Method suggests K = 5 as the optimal number "
        "of clusters for this dataset."
    )
    st.divider()

    st.subheader("Select Number of Clusters")

    k = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5
    )

    # Train model

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    df_clustered = df.copy()

    df_clustered["Cluster"] = clusters

    # Silhouette Score

    score = silhouette_score(
        X,
        clusters
    )

    st.metric(
        "Silhouette Score",
        f"{score:.3f}"
    )

    st.subheader("📊 Model Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Number of Clusters",
            k
        )

    with col2:
        st.metric(
            "Silhouette Score",
            f"{score:.3f}"
        )

    with col3:
        st.metric(
            "Customers",
            len(df_clustered)
        )

    if k == 5:
        st.success(
            "K = 5 matches the optimal value identified by "
            "the Elbow Method."
        )

    st.divider()
    # Visualization

    st.subheader("Customer Clusters")

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    scatter = ax.scatter(
        X["Annual Income (k$)"],
        X["Spending Score (1-100)"],
        c=clusters,
        cmap="viridis",
        s=70
    )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=250,
        color="red",
        label="Centroids"
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segmentation using K-Means")

    ax.legend()

    st.pyplot(fig)

    # Cluster counts

    st.subheader("Customers in Each Cluster")

    cluster_counts = (
        df_clustered["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.dataframe(
        cluster_counts.rename(
            "Number of Customers"
        ),
        use_container_width=True
    )

    st.subheader("📈 Understanding the Silhouette Score")

    st.write(
        """
        The Silhouette Score measures how well each customer
        fits within its assigned cluster compared with other
        clusters.

        A higher score generally indicates better-defined and
        more separated clusters.
        """
    )

    if score >= 0.5:
        st.success(
            f"The current Silhouette Score is {score:.3f}, "
            "indicating reasonably well-separated clusters."
        )
    elif score >= 0.25:
        st.warning(
            f"The current Silhouette Score is {score:.3f}, "
            "indicating moderate cluster separation."
        )
    else:
        st.error(
            f"The current Silhouette Score is {score:.3f}, "
            "indicating weak cluster separation."
        )
# --------------------------------------------------
# CUSTOMER SEGMENTS
# --------------------------------------------------

elif page == "Customer Segments":

    st.title("👥 Customer Segment Analysis")
    st.markdown(
        """
        This section summarizes the characteristics of each
        customer cluster and automatically assigns a meaningful
        business-oriented name based on income and spending behavior.
        """
    )

    kmeans = KMeans(
        n_clusters=5,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    df_segments = df.copy()

    df_segments["Cluster"] = kmeans.fit_predict(X)

    cluster_summary = (
        df_segments
        .groupby("Cluster")
        [["Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
    )

    st.subheader("Cluster Characteristics")

    st.dataframe(
        cluster_summary.round(2),
        use_container_width=True
    )

    st.subheader("Customer Segment Interpretation")

    segment_names = {
        0: "Regular Customers",
        1: "Target Customers",
        2: "Impulse Buyers",
        3: "Potential Customers",
        4: "Budget Customers"
    }

    for cluster, row in cluster_summary.iterrows():

        name = segment_names.get(
            cluster,
            f"Cluster {cluster}"
        )
        
        st.write(
            f"**Cluster {cluster}: {name}**"
        )

        st.write(
            f"Average Annual Income: "
            f"{row['Annual Income (k$)']:.2f} k$"
        )

        st.write(
            f"Average Spending Score: "
            f"{row['Spending Score (1-100)']:.2f}"
        )

        st.divider()
    
    # Download

    st.subheader("Download Clustered Dataset")

    csv = df_segments.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Clustered CSV",
        data=csv,
        file_name="Mall_Customers_Clustered.csv",
        mime="text/csv"
    )