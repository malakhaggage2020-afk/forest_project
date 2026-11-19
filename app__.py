import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import (COVER_NAMES, FEATURES, NUM_COLS, TARGET, WILD_NAMES,
                    build_row, load_covtype)

st.set_page_config(page_title="Forest Cover Type", page_icon="🌲", layout="wide")

DATA_PATH = next((p for p in ["covtype.data", "covtype_sample.csv.gz"] if os.path.exists(p)), None)
MODEL_PATH = "xgb_covtype.json"
METRICS_PATH = "metrics.json"


# ---------- cached loaders ----------
@st.cache_data(show_spinner="Loading data...")
def get_data(path):
    return load_covtype(path)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    m = XGBClassifier()
    m.load_model(MODEL_PATH)
    return m


@st.cache_data
def get_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def name_of(c):
    return f"{c} - {COVER_NAMES[c]}"


# ---------- data source ----------
st.title("🌲 Forest Cover Type Prediction")
st.caption("UCI Covertype dataset · 54 features · 7 forest cover classes")

if DATA_PATH:
    df = get_data(DATA_PATH)
else:
    up = st.file_uploader("Upload covtype.data", type=["data", "csv"])
    if up is None:
        st.info("Put `covtype.data` or `covtype_sample.csv.gz` next to `app.py`, or upload it here.")
        st.stop()
    df = get_data(up)

tab_data, tab_eda, tab_pred, tab_perf, tab_clu = st.tabs(
    ["📊 Data", "🔎 Exploration", "🎯 Predict", "🏆 Model performance", "🧩 Clustering"])

# ---------- 1) Data ----------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Classes", df[TARGET].nunique())

    st.subheader("Class distribution")
    dist = df[TARGET].value_counts().sort_index()
    dist.index = [name_of(i) for i in dist.index]
    st.bar_chart(dist)
    st.caption("The classes are imbalanced: Lodgepole Pine and Spruce/Fir make up most of the data, "
               "while Cottonwood/Willow is very rare. This is why macro F1 is reported next to accuracy.")

    st.subheader("Sample rows")
    st.dataframe(df.sample(10, random_state=1), width="stretch")
    st.subheader("Numeric feature summary")
    st.dataframe(df[NUM_COLS].describe().T, width="stretch")

# ---------- 2) EDA ----------
with tab_eda:
    feat = st.selectbox("Feature", NUM_COLS, index=0)
    classes = st.multiselect("Classes", sorted(COVER_NAMES), default=sorted(COVER_NAMES),
                             format_func=name_of)
    sub = df[df[TARGET].isin(classes)].sample(min(60000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in classes:
        ax.hist(sub.loc[sub[TARGET] == c, feat], bins=40, alpha=0.5,
                density=True, label=name_of(c))
    ax.set_xlabel(feat); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption("Elevation separates the classes the most, which is why tree-based models perform so well here.")

    st.subheader("Correlation between numeric features")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    corr = df[NUM_COLS].corr()
    im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(NUM_COLS))); ax2.set_yticks(range(len(NUM_COLS)))
    ax2.set_xticklabels(NUM_COLS, rotation=90, fontsize=7)
    ax2.set_yticklabels(NUM_COLS, fontsize=7)
    fig2.colorbar(im)
    st.pyplot(fig2)

# ---------- 3) Predict ----------
with tab_pred:
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found. Run `python train_model.py` first.")
    else:
        model = get_model()
        ranges = {c: (int(df[c].min()), int(df[c].max()), int(df[c].median())) for c in NUM_COLS}

        # session defaults
        for c in NUM_COLS:
            st.session_state.setdefault(f"in_{c}", ranges[c][2])
        st.session_state.setdefault("in_wild", 1)
        st.session_state.setdefault("in_soil", 1)
        st.session_state.setdefault("true_label", None)

        def load_random_row():
            r = df.sample(1).iloc[0]
            for c in NUM_COLS:
                st.session_state[f"in_{c}"] = int(r[c])
            st.session_state["in_wild"] = int(
                [i for i in range(1, 5) if r[f"Wilderness_Area{i}"] == 1][0])
            st.session_state["in_soil"] = int(
                [i for i in range(1, 41) if r[f"Soil_Type{i}"] == 1][0])
            st.session_state["true_label"] = int(r[TARGET])

        left, right = st.columns([3, 2])
        with left:
            st.button("🎲 Load a random real sample", on_click=load_random_row)
            cols = st.columns(2)
            vals = {}
            for i, c in enumerate(NUM_COLS):
                lo, hi, _ = ranges[c]
                vals[c] = cols[i % 2].slider(c, lo, hi, key=f"in_{c}")
            wild = st.selectbox("Wilderness area", [1, 2, 3, 4], key="in_wild",
                                format_func=lambda i: f"{i} - {WILD_NAMES[i]}")
            soil = st.selectbox("Soil type", list(range(1, 41)), key="in_soil",
                                format_func=lambda i: f"Soil type {i}")

        with right:
            x = build_row(vals, wild, soil)
            # match the column names the saved model was trained with (e.g. 0..53)
            model_cols = model.get_booster().feature_names
            if model_cols and list(x.columns) != list(model_cols):
                x.columns = model_cols
            proba = model.predict_proba(x)[0]
            pred = int(np.argmax(proba)) + 1
            st.subheader("Prediction")
            st.success(f"**{name_of(pred)}**  ·  confidence {proba.max():.1%}")
            if st.session_state["true_label"] is not None:
                t = st.session_state["true_label"]
                (st.info if t == pred else st.error)(f"Actual label of the loaded sample: {name_of(t)}")
            st.bar_chart(pd.Series(proba, index=[COVER_NAMES[i] for i in range(1, 8)]))

# ---------- 4) Performance ----------
with tab_perf:
    if not os.path.exists(METRICS_PATH):
        st.warning("Run `python train_model.py` to generate the metrics.")
    else:
        m = get_metrics()
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Macro F1", f"{m['macro_f1']:.3f}")
        c3.metric("Training time", f"{m['train_seconds']:.0f}s")

        rep = pd.DataFrame(m["report"]).T.iloc[:7][["precision", "recall", "f1-score", "support"]]
        rep.index = [name_of(i + 1) for i in range(7)]
        st.subheader("Per-class metrics")
        st.dataframe(rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                       "f1-score": "{:.2f}", "support": "{:.0f}"}),
                     width="stretch")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Confusion matrix (row-normalized)")
            cm = np.array(m["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(7)); ax.set_yticks(range(7))
            ax.set_xticklabels(range(1, 8)); ax.set_yticklabels(range(1, 8))
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
            st.pyplot(fig)
        with cc2:
            st.subheader("Top 10 feature importances")
            fi = pd.Series(m["feature_importance"]).sort_values(ascending=False).head(10)
            fi.index = [FEATURES[int(i)] if str(i).isdigit() else i for i in fi.index]
            st.bar_chart(fi)

# ---------- 5) Clustering ----------
with tab_clu:
    st.write("Unsupervised: the labels are **not** used for clustering, only to evaluate it afterwards (ARI).")
    c1, c2 = st.columns(2)
    n = c1.slider("Sample size", 2000, 20000, 8000, step=1000)
    k = c2.slider("Number of clusters (k)", 2, 10, 7)

    @st.cache_data(show_spinner="Clustering...")
    def run_clustering(n, k):
        s = df.sample(n, random_state=42)
        X = StandardScaler().fit_transform(s[FEATURES])
        ks = list(range(2, 11))
        inertia = [KMeans(n_clusters=i, n_init=5, random_state=42).fit(X).inertia_ for i in ks]
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(n, 5000), random_state=42)
        ari = adjusted_rand_score(s[TARGET], labels)
        xy = PCA(n_components=2, random_state=42).fit_transform(X)
        return ks, inertia, labels, sil, ari, xy

    ks, inertia, labels, sil, ari, xy = run_clustering(n, k)
    m1, m2 = st.columns(2)
    m1.metric("Silhouette score", f"{sil:.3f}")
    m2.metric("Adjusted Rand Index vs true classes", f"{ari:.3f}")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ks, inertia, marker="o"); ax.set_xlabel("k"); ax.set_ylabel("Inertia")
        ax.set_title("Elbow method")
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=4, cmap="tab10")
        ax.set_title(f"KMeans (k={k}) on PCA 2D")
        st.pyplot(fig)
    st.caption("Low ARI is expected: the 7 cover types are not naturally separated into compact clusters, "
               "so unsupervised clustering cannot recover them well.")
