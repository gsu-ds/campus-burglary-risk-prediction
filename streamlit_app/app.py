import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Atlanta Burglary Risk", layout="wide")

st.title("Spatiotemporal Forecasting of Burglary Risk in Atlanta")
st.write(
    "This dashboard explores burglary and larceny incidents reported by the "
    "Atlanta Police Department between 2020 and 2024, grouped by Neighborhood "
    "Planning Units (NPUs)."
)
st.caption(
    "Use the filters on the left to switch between training years (2020–2023) "
    "and testing year (2024), and to focus on specific NPUs."
)
with st.expander("What this dashboard shows"):
    st.write(
        "- **Data source:** Atlanta Police Department Open Data, filtered to burglary and "
        "larceny incidents (2020–2024).\n"
        "- **Spatial unit:** Neighborhood Planning Units (NPUs) across Atlanta.\n"
        "- **Temporal unit:** Incidents aggregated at an hourly level.\n"
        "- **Modeling plan:** Use 2020–2023 as training data and 2024 as a hold-out test "
        "set for forecasting models (Random Forest, XGBoost, Prophet)."
    )

@st.cache_data
def load_data():
    path = Path("data/crime_dataset.csv")
    df = pd.read_csv(path)
    
    if "report_date" in df.columns:
        df["datetime"] = pd.to_datetime(df["report_date"])
    elif "ReportDate" in df.columns:
        df["datetime"] = pd.to_datetime(df["ReportDate"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError(
            "Could not find a date column. Expected one of: report_date, "
            "ReportDate, datetime."
        )
    
    if "npu" not in df.columns:
        if "npu_label" in df.columns:
            df["npu"] = df["npu_label"]
        else:
            # Fallback if you only have citywide data
            df["npu"] = "All"

    if "lat" not in df.columns:
        if "latitude" in df.columns:
            df["lat"] = df["latitude"]
        elif "Latitude" in df.columns:
            df["lat"] = df["Latitude"]
        else:
            df["lat"] = pd.NA

    if "lon" not in df.columns:
        if "longitude" in df.columns:
            df["lon"] = df["longitude"]
        elif "Longitude" in df.columns:
            df["lon"] = df["Longitude"]
        else:
            df["lon"] = pd.NA

    df["year"] = df["datetime"].dt.year

    return df

    
@st.cache_data
def load_forecasts():
    path = Path("data/forecast_results_2024.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df

df = load_data()

st.sidebar.header("Filters")
st.sidebar.caption(
    "Training set = 2020–2023. Testing set = 2024. "
    "All visualizations update based on these filters."
)

year_slice = st.sidebar.radio(
    "Year range",
    ["All years", "Training (2020–2023)", "Testing (2024)"],
)

if year_slice == "Training (2020–2023)":
    df_year = df[(df["year"] >= 2020) & (df["year"] <= 2023)]
elif year_slice == "Testing (2024)":
    df_year = df[df["year"] == 2024]
else:
    df_year = df.copy()

npu_options = sorted(df_year["npu"].dropna().unique())
selected_npus = st.sidebar.multiselect(
    "Neighborhood Planning Units (NPUs)",
    npu_options,
    default=npu_options,
)

df_view = df_year[df_year["npu"].isin(selected_npus)]

st.sidebar.write(f"Rows in current view: {len(df_view):,}")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Incidents (filtered)", f"{len(df_view):,}")
with col2:
    st.metric("NPUs selected", df_view["npu"].nunique())
with col3:
    st.metric("Year range in view", f"{df_view['year'].min()}–{df_view['year'].max()}" if not df_view.empty else "–")

st.subheader("Raw data preview (after filters)")
st.caption(
    "First few rows of the filtered dataset. Each row is a reported incident."
)
st.dataframe(df_view.head(), use_container_width=True)

st.divider()

st.subheader("Incidents over time")

agg_level = st.radio(
    "Aggregation level",
    ["Hourly", "Daily"],
    horizontal=True,
)

if df_view.empty:
    st.info("No data for this filter combo.")
else:
    if agg_level == "Hourly":
        ts = (
            df_view.assign(datetime_bucket=df_view["datetime"].dt.floor("H"))
            .groupby("datetime_bucket")
            .size()
            .reset_index(name="incident_count")
            .set_index("datetime_bucket")
        )
    else:
        ts = (
            df_view.assign(datetime_bucket=df_view["datetime"].dt.floor("D"))
            .groupby("datetime_bucket")
            .size()
            .reset_index(name="incident_count")
            .set_index("datetime_bucket")
        )

    if ts.empty:
        st.info("No data for this filter combo.")
    else:
        st.line_chart(ts)
        st.caption(
            f"{agg_level} incident counts for the selected years and NPUs. "
            "Daily aggregation smooths short-term noise and highlights broader trends."
        )

st.subheader("Incidents over time (hourly)")

if df_view.empty:
    st.info("No data for this filter combo.")
else:
    ts = (
        df_view.assign(datetime_hour=df_view["datetime"].dt.floor("H"))
        .groupby("datetime_hour")
        .size()
        .reset_index(name="incident_count")
        .set_index("datetime_hour")
    )

    if ts.empty:
        st.info("No data for this filter combo.")
    else:
        st.line_chart(ts)
        st.caption(
            "Hourly incident counts for the selected years and NPUs. "
            "This shows historical patterns we’ll use to train and test "
            "forecasting models."
        )

st.divider()

st.subheader("Incidents by NPU (current filters)")

if df_view.empty:
    st.info("No NPU data available for this filter combo.")
else:
    npu_counts = (
        df_view.groupby("npu")
        .size()
        .reset_index(name="incident_count")
        .sort_values("incident_count", ascending=False)
    )

    if npu_counts.empty:
        st.info("No NPU data available for this filter combo.")
    else:
        st.bar_chart(npu_counts.set_index("npu")["incident_count"])
        st.caption(
            "Total incidents per NPU for the selected years and filters. "
            "Higher bars indicate higher historical risk."
        )

st.divider()

st.subheader("Spatial view of incidents (points)")

if df_view.empty:
    st.info("No locations to show for these filters.")
else:
    map_df = (
        df_view[["lat", "lon"]]
        .dropna()
        .rename(columns={"lat": "latitude", "lon": "longitude"})
    )

    if map_df.empty:
        st.info("No locations to show for these filters.")
    else:
        if len(map_df) > 5000:
            map_df = map_df.sample(5000, random_state=42)

        st.map(map_df)
        st.caption(
            "Each point shows an incident location for the current filters. "
            "Zoom in to explore local hot spots."
        )

st.divider()

st.subheader("Spatial view of incidents (by NPU)")

if df_view.empty:
    st.info("No spatial data available for this filter combo.")
else:
    map_data = (
        df_view.groupby("npu")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            incident_count=("npu", "size"),
        )
        .reset_index()
        .dropna(subset=["lat", "lon"])
    )

    if map_data.empty:
        st.info("No spatial data available for this filter combo.")
    else:
        max_count = max(map_data["incident_count"].max(), 1)
        map_data["risk_level"] = map_data["incident_count"] / max_count

        center_lat = map_data["lat"].mean()
        center_lon = map_data["lon"].mean()

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position="[lon, lat]",
                    get_radius="200 + risk_level * 1200",
                    get_fill_color="[255, 140, 0, 160]",
                    pickable=True,
                )
            ],
            tooltip={"text": "NPU {npu}\nIncidents: {incident_count}"},
        )

        st.pydeck_chart(deck)
        st.caption(
            "Each circle represents an NPU. Larger circles = more incidents "
            "for the current time range and NPUs. This highlights higher-risk areas."
        )

#Forecast Section UI
st.divider()
st.subheader("Forecasting")

forecast_df = load_forecasts()

if forecast_df is None:
    st.info(
        "Forecast results file not found. Once we export baseline forecasts to "
        "`data/forecast_results_2024.csv`, this section will show actual vs. "
        "predicted incident counts for 2024."
    )
else:

    st.info(
        "We forecast daily burglary and larceny incidents at the NPU level using "
        "a baseline time-series model. For each NPU, we predict a given day's "
        "incidents using a 7-day rolling average of past incidents. The dashboard "
        "summarizes forecast accuracy across NPUs and then lets you drill into a "
        "single NPU to compare baseline predictions with actual 2024 incident "
        "counts. Future work can replace this baseline with more complex models "
        "such as Random Forest, XGBoost, or Prophet trained on 2020–2023."
    )


    y_true_all = forecast_df["actual"].to_numpy()

    y_pred_all = forecast_df["rf_pred"].to_numpy()

    overall_rmse = float(np.sqrt(((y_true_all - y_pred_all) ** 2).mean()))
    overall_mae = float(np.abs(y_true_all - y_pred_all).mean())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("NPUs covered", forecast_df["npu"].nunique())
    with c2:
        st.metric("Baseline RMSE", f"{overall_rmse:.2f}")
    with c3:
        st.metric("Baseline MAE", f"{overall_mae:.2f}")

    st.divider()


    tab_summary, tab_detail = st.tabs(["NPU summary", "Daily forecast for one NPU"])

    with tab_summary:
        npu_stats = []
        for npu_val, g in forecast_df.groupby("npu"):
            y_true = g["actual"].to_numpy()
            y_pred = g["rf_pred"].to_numpy()
            rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
            mae = float(np.abs(y_true - y_pred).mean())
            mean_inc = float(y_true.mean())
            npu_stats.append(
                {
                    "NPU": npu_val,
                    "Mean daily incidents": mean_inc,
                    "RMSE": rmse,
                    "MAE": mae,
                }
            )

        stats_df = pd.DataFrame(npu_stats).sort_values("RMSE")
        st.subheader("Baseline forecast accuracy by NPU")
        st.caption(
            "Lower RMSE/MAE indicate better forecast performance. "
            "This table lets you quickly compare how the baseline model performs "
            "across NPUs."
        )
        st.dataframe(stats_df.set_index("NPU").round(2), use_container_width=True)


    with tab_detail:
        st.subheader("Baseline forecasts vs. actual (2024)")

        available_npus = sorted(forecast_df["npu"].dropna().unique())
        if not available_npus:
            st.info("No forecast NPUs available in the results file.")
        else:
            default_npu = (
                selected_npus[0]
                if selected_npus and selected_npus[0] in available_npus
                else available_npus[0]
            )

            npu_choice = st.selectbox(
                "NPU for forecast view",
                available_npus,
                index=available_npus.index(default_npu),
            )

            df_npu = (
                forecast_df[forecast_df["npu"] == npu_choice]
                .sort_values("datetime")
            )

            if df_npu.empty or "rf_pred" not in df_npu.columns:
                st.info("No forecast data for this NPU.")
            else:
                plot_df = df_npu[["datetime", "actual", "rf_pred"]].set_index("datetime")
                plot_df = plot_df.rename(columns={"rf_pred": "predicted"})

                st.line_chart(plot_df)
                st.caption(
                    "Actual vs. predicted daily incident counts for 2024 in the "
                    "selected NPU using the baseline 7-day rolling-average model."
                )

                y_true = df_npu["actual"].to_numpy()
                y_pred = df_npu["rf_pred"].to_numpy()

                rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
                mae = float(np.abs(y_true - y_pred).mean())

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("NPU-level RMSE", f"{rmse:.2f}")
                with c2:
                    st.metric("NPU-level MAE", f"{mae:.2f}")