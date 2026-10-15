
import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA


st.set_page_config(page_title="Iris K-Means Dashboard", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Clustering Dashboard")
st.caption("Explore the Iris dataset and predict K-Means clusters with your trained model.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_kmeans_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "Iris.csv")

@st.cache_resource
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    bundle = load_bundle(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]
except Exception as e:
    st.error(f"Could not load model file: {e}")
    st.info("Place iris_kmeans_model.pkl in the same folder as app.py.")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load Iris.csv: {e}")
    st.info("Place Iris.csv in the same folder as app.py.")
    st.stop()

missing = [c for c in features if c not in df.columns]
if missing:
    st.error("Dataset is missing required feature columns: " + ", ".join(missing))
    st.stop()

X = df[features].copy()
X_scaled = scaler.transform(X)
df["Cluster"] = model.predict(X_scaled).astype(str)

# Sidebar controls
st.sidebar.header("🔎 Dashboard Filters")
cluster_options = sorted(df["Cluster"].unique(), key=lambda x: int(x))
selected_clusters = st.sidebar.multiselect(
    "Choose clusters", cluster_options, default=cluster_options
)
show_data = st.sidebar.checkbox("Show filtered data table", value=True)

filtered = df[df["Cluster"].isin(selected_clusters)].copy()

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total flowers", f"{len(df):,}")
c2.metric("Clusters", f"{df['Cluster'].nunique()}")
c3.metric("Filtered flowers", f"{len(filtered):,}")
c4.metric("Features", f"{len(features)}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Cluster distribution")
    counts = filtered["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Count")
    if len(counts):
        fig = px.bar(counts, x="Cluster", y="Count", color="Cluster", text="Count",
                     title="Number of flowers in each cluster")
        fig.update_layout(showlegend=False, xaxis_title="K-Means cluster", yaxis_title="Flower count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one cluster in the sidebar.")

with right:
    st.subheader("Feature relationship")
    x_feature = st.selectbox("X-axis feature", features, index=2)
    y_feature = st.selectbox("Y-axis feature", features, index=3)
    if len(filtered):
        fig = px.scatter(filtered, x=x_feature, y=y_feature, color="Cluster",
                         hover_data=features, title=f"{x_feature} vs {y_feature}")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Cluster view (PCA)")
if len(filtered) >= 2:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaler.transform(filtered[features]))
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=filtered.index)
    pca_df["Cluster"] = filtered["Cluster"].values
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                     title="Flowers projected into two dimensions",
                     labels={"PC1": "Principal component 1", "PC2": "Principal component 2"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least two flowers to display the PCA chart.")

st.subheader("🌼 Try a new flower")
st.write("Enter the four measurements in centimeters to predict its cluster.")
with st.form("prediction_form"):
    p1, p2 = st.columns(2)
    with p1:
        sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    with p2:
        petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    submitted = st.form_submit_button("Predict cluster", type="primary")

if submitted:
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=features)
    prediction = model.predict(scaler.transform(sample))[0]
    st.success(f"Predicted K-Means cluster: {prediction}")
    st.caption("Cluster IDs are group labels, not confirmed Iris species names.")

if show_data:
    st.subheader("Dataset preview")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.caption("Model: K-Means clustering • Features are standardized before prediction.")
