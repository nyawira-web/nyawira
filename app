import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Medical Health Statistics Explorer", layout="wide")

MEASURES = {
    "BMI (kg/m²)": "BMI_kg_m2",
    "Systolic BP (mmHg)": "Systolic_BP_mmHg",
    "Diastolic BP (mmHg)": "Diastolic_BP_mmHg",
    "Fasting glucose (mg/dL)": "Fasting_Glucose_mg_dL",
    "Total cholesterol (mg/dL)": "Total_Cholesterol_mg_dL",
    "Resting heart rate (bpm)": "Resting_Heart_Rate_bpm",
}
AGE_BINS = [17, 29, 39, 49, 59, 80]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-80"]


@st.cache_data
def load():
    df = pd.read_csv("Synthetic_Medical_Health_Screening_Dataset.csv")
    df["Age_group"] = pd.cut(df["Age_years"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def stats_table(s: pd.Series) -> dict:
    q1, q3 = s.quantile([0.25, 0.75])
    modes = s.round(0).mode()
    return {
        "n": int(s.count()),
        "Mean": s.mean(),
        "Median": s.median(),
        "Mode (rounded)": modes.iloc[0] if len(modes) else np.nan,
        "Min": s.min(),
        "Max": s.max(),
        "Range": s.max() - s.min(),
        "Variance": s.var(ddof=1),
        "Std dev": s.std(ddof=1),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "CV (%)": s.std(ddof=1) / s.mean() * 100,
        "Skewness": s.skew(),
    }


def interpret(label: str, s: pd.Series) -> str:
    t = stats_table(s)
    q1, q3, iqr = t["Q1"], t["Q3"], t["IQR"]
    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    gap = t["Mean"] - t["Median"]
    sk = t["Skewness"]
    if abs(sk) < 0.2:
        shape = "approximately symmetric"
    elif sk > 0:
        shape = "mildly right-skewed (longer upper tail)"
    else:
        shape = "mildly left-skewed (longer lower tail)"
    cv = t["CV (%)"]
    spread = "low" if cv < 12 else "moderate" if cv < 20 else "high"
    return (
        f"For **{label}**, the mean ({t['Mean']:.2f}) and median ({t['Median']:.2f}) differ by "
        f"{gap:+.2f}, and the distribution is {shape} (skewness {sk:.2f}). "
        f"The standard deviation is {t['Std dev']:.2f} and the IQR is {iqr:.2f}, giving a coefficient "
        f"of variation of {cv:.1f}% ({spread} relative variability). "
        f"By the 1.5 × IQR rule, {n_out} value(s) are statistical outliers. "
        "When outliers or skewness are present, the median is the more robust summary of a typical value."
    )


df = load()
st.title("Medical Health Screening: Statistical Explorer")
st.caption("Simulated dataset of 500 individuals. Results are statistical observations only, "
           "not clinical diagnoses or treatment advice.")

# Sidebar controls
st.sidebar.header("Controls")
label = st.sidebar.selectbox("Health measurement", list(MEASURES))
col = MEASURES[label]
sexes = st.sidebar.multiselect("Sex", ["Female", "Male"], default=["Female", "Male"])
ages = st.sidebar.multiselect("Age group", AGE_LABELS, default=AGE_LABELS)

fdf = df[df["Sex"].isin(sexes) & df["Age_group"].isin(ages)]
if fdf.empty:
    st.warning("No data for the selected filters.")
    st.stop()

s = fdf[col]
tab1, tab2, tab3, tab4 = st.tabs(["Summary statistics", "Distribution", "Group comparison", "Data"])

with tab1:
    t = stats_table(s)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{t['Mean']:.2f}")
    c2.metric("Median", f"{t['Median']:.2f}")
    c3.metric("Mode (rounded)", f"{t['Mode (rounded)']:.0f}")
    c4.metric("n", t["n"])
    st.subheader("Central tendency and dispersion")
    tbl = pd.DataFrame(t, index=[label]).T.drop("n")
    tbl.columns = ["Value"]
    st.dataframe(tbl.round(2))
    st.subheader("Interpretation")
    st.write(interpret(label, s))
    st.info("The mode of continuous data is only indicative: it is computed after rounding to whole units.")

with tab2:
    c1, c2 = st.columns(2)
    bins = st.sidebar.slider("Histogram bins", 10, 60, 25)
    fig_h = px.histogram(fdf, x=col, nbins=bins, marginal=None, title=f"Histogram of {label}")
    fig_h.add_vline(x=s.mean(), line_dash="dash", line_color="red", annotation_text="Mean")
    fig_h.add_vline(x=s.median(), line_dash="dot", line_color="green", annotation_text="Median", annotation_position="bottom")
    c1.plotly_chart(fig_h)
    fig_b = px.box(fdf, y=col, points="outliers", title=f"Box plot of {label}")
    c2.plotly_chart(fig_b)

with tab3:
    by = st.radio("Compare by", ["Age group", "Sex", "Age group and sex"], horizontal=True)
    if by == "Age group":
        g, color = "Age_group", None
    elif by == "Sex":
        g, color = "Sex", None
    else:
        g, color = "Age_group", "Sex"
    keys = [g] + ([color] if color else [])
    summ = (fdf.groupby(keys, observed=True)[col]
            .agg(n="count", Mean="mean", Median="median", SD="std",
                 IQR=lambda x: x.quantile(.75) - x.quantile(.25))
            .reset_index())
    summ["CV (%)"] = summ["SD"] / summ["Mean"] * 100
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(summ, x=g, y="Mean", color=color, barmode="group",
                           category_orders={"Age_group": AGE_LABELS},
                           title=f"Mean {label} by {by.lower()}"))
    c2.plotly_chart(px.box(fdf, x=g, y=col, color=color,
                           category_orders={"Age_group": AGE_LABELS},
                           title=f"Spread of {label} by {by.lower()}"))
    st.dataframe(summ.round(2))
    sub = summ.groupby(g, observed=True)["Mean"].mean()
    hi, lo = sub.idxmax(), sub.idxmin()
    st.write(f"Mean {label} is highest in **{hi}** ({sub.max():.2f}) and lowest in **{lo}** ({sub.min():.2f}). "
             "Comparing the SD and IQR columns shows whether the spread also changes between groups "
             "or whether the whole distribution simply shifts.")

with tab4:
    st.dataframe(fdf)

st.divider()
st.caption("Limitations: the data are simulated and cannot represent a real population. "
           "Statistical outliers are not diagnoses, and nothing here should be used to diagnose "
           "individuals or recommend treatment.")