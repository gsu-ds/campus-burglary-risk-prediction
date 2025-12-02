import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Atlanta Burglary Risk", layout="wide")

@st.cache_data
def load_data():
    path = Path("data/processed/apd/target_crimes.csv")
    df = pd.read_csv(path)

    for col in ["datetime", "incident_datetime", "report_date", "ReportDate", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.rename(columns={col: "datetime"})
            break
    if "datetime" not in df.columns:
        raise ValueError("Could not find a datetime column in target_crimes.csv")

    if "npu" not in df.columns:
        if "npu_right" in df.columns or "npu_left" in df.columns:
            df["npu"] = pd.Series(pd.NA, index=df.index)
            if "npu_right" in df.columns:
                df["npu"] = df["npu_right"]
            if "npu_left" in df.columns:
                df["npu"] = df["npu"].fillna(df["npu_left"])
        else:
            df["npu"] = "Unknown"

    if "time_block" not in df.columns:
        for c in ["time_block", "hour_block", "time_bin", "time_block_label"]:
            if c in df.columns:
                df = df.rename(columns={c: "time_block"})
                break

    if "lat" not in df.columns:
        if "latitude" in df.columns:
            df["lat"] = df["latitude"]
        elif "Latitude" in df.columns:
            df["lat"] = df["Latitude"]

    if "lon" not in df.columns:
        if "longitude" in df.columns:
            df["lon"] = df["longitude"]
        elif "Longitude" in df.columns:
            df["lon"] = df["Longitude"]

    df["year"] = df["datetime"].dt.year
    df["date"] = df["datetime"].dt.date

    return df

df = load_data()

st.title("Spatiotemporal Forecasting of Burglary Risk in Atlanta")

nav1, nav2, nav3, nav4, nav5 = st.columns([2, 1, 1, 1, 1])
with nav2:
    st.markdown("[Project Abstract](https://docs.google.com/document/d/1PreU-Ik2mYIIW4Ja753anvFQFjZGCTmt4FpvtWkVoes/edit?tab=t.5o4kiuwqlscf#heading=h.b7u8zxwyyq9c)")
with nav3:
    st.markdown("[Data & Sources](#data--sources)")
with nav4:
    # TODO: replace with real link when your slides are final
    st.markdown("[Final Presentation](https://docs.google.com/presentation/d/1OrThUntGbi8fWF3Qqyw-xHaza5_X8Qha0KVonl19IV8/edit?slide=id.gc6f80d1ff_0_0#slide=id.gc6f80d1ff_0_0)")
with nav5:
    st.markdown(
        "[GitHub Repo](https://github.com/gsu-ds/campus-burglary-risk-prediction)"
    )

st.markdown("---")

st.markdown("## About", help=None)

st.markdown(
    """
This dashboard explores **burglary and larceny risk around Atlanta’s major universities**
using Atlanta Police Department (APD) incident data enriched with **time, weather, and
spatial features**.

We aggregate incidents from **2021–2025** at the **Neighborhood Planning Unit (NPU)**
and **time-of-day block** level, then use forecasting models to estimate relative risk
across space and time. The goal is to help stakeholders:
- Identify **higher-risk NPUs** and time windows,
- Compare **historical patterns** across campuses and neighborhoods,
- Support **data-informed decisions** about patrols, outreach, and safety resources.
"""
)

st.caption(
    "Use the controls below to explore trends by NPU, time of day, and year. "
    "Model details and data sources are available in the tabs and navigation links."
)

st.markdown("---")

total_crimes = len(df)
years = f"{df['year'].min()}–{df['year'].max()}"
npu_count = df["npu"].nunique() if "npu" in df.columns else 0

m1, m2, m3 = st.columns(3)
m1.metric("Crimes tracked", f"{total_crimes:,}")
m2.metric("Years covered", years)
m3.metric("NPUs analyzed", npu_count)

st.markdown("### Explore forecasted risk by NPU & time of day")

st.sidebar.header("Filters")
st.sidebar.caption("Filter the dataset by year range and NPU(s).")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

df_year = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

npu_options = (
    sorted(df_year["npu"].dropna().unique()) if "npu" in df_year.columns else []
)
selected_npus = st.sidebar.multiselect(
    "Neighborhood Planning Units (NPUs)",
    npu_options,
    default=npu_options,
)

if selected_npus:
    df_view = df_year[df_year["npu"].isin(selected_npus)].copy()
else:
    df_view = df_year.copy()

st.sidebar.write(f"Rows in current view: {len(df_view):,}")

time_block_options = (
    sorted(df["time_block"].dropna().unique())
    if "time_block" in df.columns
    else []
)

has_npu = "npu" in df.columns

cta_left, cta_right = st.columns([2, 1])

if has_npu:
    with cta_left:
        selected_npu_cta = st.selectbox(
            "Focus NPU",
            npu_options if npu_options else ["(no NPU column)"],
            index=0 if npu_options else 0,
        )
else:
    selected_npu_cta = None
    with cta_left:
        st.selectbox(
            "Focus NPU",
            ["(NPU column not found in dataset)"],
            index=0,
            disabled=True,
        )

with cta_right:
    if time_block_options:
        selected_tb = st.selectbox(
            "Time-of-day block",
            options=time_block_options,
            format_func=lambda x: str(x),
        )
    else:
        selected_tb = None

st.caption(
    "Choose an NPU and time-of-day block to explore historical patterns and future risk."
)

if has_npu and selected_npu_cta and selected_tb is not None:
    subset_cta = df[
        (df["npu"] == selected_npu_cta) & (df["time_block"] == selected_tb)
    ]

    if not subset_cta.empty:
        daily_counts = subset_cta.groupby("date").size()
        avg_incidents = daily_counts.mean()
        st.info(
            f"On average, NPU **{selected_npu_cta}** sees about "
            f"**{avg_incidents:0.2f} incidents per day** in the "
            f"**{selected_tb}** time block over the selected years."
        )
    else:
        st.info("No incidents found for this NPU/time block combination.")
elif not has_npu:
    st.info("NPU column not found in the dataset, so NPU-specific views are disabled.")


    if not subset_cta.empty:
        # Simple baseline-ish metric: average incidents in that NPU x time block
        daily_counts = subset_cta.groupby("date").size()
        avg_incidents = daily_counts.mean()
        st.info(
            f"On average, NPU **{selected_npu_cta}** sees about "
            f"**{avg_incidents:0.2f} incidents per day** in the "
            f"**{selected_tb}** time block over the selected years."
        )
    else:
        st.info("No incidents found for this NPU/time block combination.")

st.divider()

tab_explore, tab_models, tab_sources, tab_faq = st.tabs(
    ["📊 Explore Data", "🧠 How the Model Works", "📚 Data & Sources", "❓ FAQ"]
)

with tab_explore:
    st.subheader("Raw data preview (after filters)")
    st.caption("First few rows of the filtered dataset.")
    st.dataframe(df_view.head(), use_container_width=True)

    st.divider()

    st.subheader("Incidents over time")

    if df_view.empty:
        st.info("No data for this filter combo.")
    else:
        # Daily counts + 7-day rolling avg to reduce noise
        ts = (
            df_view.groupby("date")
            .size()
            .rename("incidents")
            .sort_index()
        )
        ts_rolling = ts.rolling(7, center=True).mean()

        ts_df = pd.DataFrame(
            {"Daily incidents": ts, "7-day rolling average": ts_rolling}
        )
        st.line_chart(ts_df)
        st.caption(
            "Daily incident counts for the selected years and NPUs. "
            "The 7-day rolling average smooths hourly noise and highlights trends."
        )

    st.divider()

    st.subheader("Incidents by NPU (current filters)")

    if df_view.empty or "npu" not in df_view.columns:
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

    st.subheader("Spatial view of incidents (by NPU)")

    if df_view.empty or "lat" not in df_view.columns or "lon" not in df_view.columns:
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
                "for the current time range. This highlights higher-risk areas."
            )

with tab_models:
    st.markdown("### How the forecasts are generated")
    st.markdown(
        """
We forecast burglary and larceny risk for each NPU and time-of-day block using a mix of
simple baselines and more expressive models.

**Baselines**

- **24-hour seasonal naïve:** uses the incident count from exactly 24 hours ago as the forecast.
- **7-day seasonal naïve (168-hour):** uses the incident count from the same hour one week ago.

These give us simple yardsticks to compare against.

**Models**

We then train several models on time, weather, and spatial features:

- **XGBoost (Poisson):** gradient-boosted trees tailored to count data.
- **CatBoost (Poisson):** handles categorical features like NPU and time block efficiently.
- **LightGBM (Poisson):** a fast gradient-boosted tree model optimized for large datasets.
- **Zero-Inflated Poisson (ZIP):** a statistical model that accounts for many zero-incident hours.
- **Prophet:** a decomposable time-series model that captures trend and seasonality.

To respect the time structure, we use **rolling cross-validation**:
we train on earlier years and validate on future months (for example,
train on 2021, validate on early 2022; then extend to 2021–2022 and validate on late 2022, etc.),
with a final hold-out test on 2024.
        """
    )

with tab_sources:
    st.markdown("## Data & Sources")
    st.markdown(
        """
- **Incident data:** Atlanta Police Department (APD) burglary & larceny reports.
- **Weather data:** Hourly and daily weather from Open-Meteo for the Atlanta area.
- **Spatial features:** NPUs, APD Zones, campus footprints, neighborhoods, and cities
  from local shapefiles.

All of this is processed into `data/processed/apd/target_crimes.csv`, which aggregates
time, weather, and spatial features for each NPU and time-of-day block.
        """
    )

with tab_faq:
    st.markdown("### FAQ")
    st.markdown(
        """
**What is an NPU?**  
Neighborhood Planning Units (NPUs) are community planning districts defined by the City of Atlanta.
We forecast risk at the NPU level to align with how local stakeholders make decisions.

**What does 'risk' mean here?**  
We use historical burglary and larceny counts as a proxy for relative risk.
Higher forecasted counts indicate times and locations where incidents are more likely.

**Can these forecasts be used to predict individual crimes?**  
No. The goal is *strategic* insight (e.g., where/when risk is higher) rather than
predicting specific incidents or individuals.
        """
    )
    
st.markdown("---")
st.caption(
    "Built by the GSU Data Science Capstone Group 3 (Fall 2025). Data: APD burglary & larceny reports, weather from Open-Meteo."
)
#Mapbox
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN")

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=0,
    ),
    layers=[ ... ],
    tooltip={"text": "NPU {npu}\nIncidents: {incident_count}"},
    mapbox_key=MAPBOX_TOKEN,
)
