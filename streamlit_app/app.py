import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Atlanta Burglary Risk", layout="wide")

st.title("Spatiotemporal Forecasting of Burglary Risk in Atlanta")
st.write(
    "Interactive dashboard using Atlanta Police Department burglary & larceny data (NPUs, 2020–2024)."
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_full_atl_v8.csv", parse_dates=["ReportDate"])

    df = df.rename(columns={
        "ReportDate": "datetime",
        "npu_label": "npu",
        "Latitude": "lat",
        "Longitude": "lon",
    })

    df["year"] = df["datetime"].dt.year

    return df


df = load_data()

st.sidebar.header("Filters")

# Year slice: all / training / testing
year_slice = st.sidebar.radio(
    "Year range",
    ["All years", "Training (2020–2023)", "Testing (2024)"]
)

if year_slice == "Training (2020–2023)":
    df_year = df[(df["year"] >= 2020) & (df["year"] <= 2023)]
elif year_slice == "Testing (2024)":
    df_year = df[df["year"] == 2024]
else:
    df_year = df.copy()

# NPU filter
npu_options = sorted(df_year["npu"].dropna().unique())
selected_npus = st.sidebar.multiselect(
    "Neighborhood Planning Units (NPUs)",
    npu_options,
    default=npu_options
)

df_view = df_year[df_year["npu"].isin(selected_npus)]

st.sidebar.write(f"Rows in current view: {len(df_view):,}")

st.subheader("Raw data preview (after filters)")
st.dataframe(df_view.head(), use_container_width=True)

st.subheader("Incidents over time (hourly)")

ts = (
    df_view
    .assign(datetime_hour=df_view["datetime"].dt.floor("H"))  # round down to the hour
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
        "Each point = number of incidents in that hour."
    )

st.subheader("Incidents by NPU (current filters)")

npu_counts = (
    df_view
    .groupby("npu")
    .size()
    .reset_index(name="incident_count")
    .sort_values("incident_count", ascending=False)
)

if npu_counts.empty:
    st.info("No NPU data available for this filter combo.")
else:
    st.bar_chart(
        npu_counts.set_index("npu")["incident_count"]
    )
    st.caption("Total incidents per NPU for the selected years and filters.")

st.subheader("Spatial view of incidents (by NPU)")

map_data = (
    df_view
    .dropna(subset=["lat", "lon", "npu"])
    .groupby("npu", as_index=False)
    .agg({
        "lat": "mean",
        "lon": "mean"
    })
)

counts = (
    df_view
    .groupby("npu")
    .size()
    .reset_index(name="incident_count")
)

map_data = map_data.merge(counts, on="npu", how="left")

if map_data.empty:
    st.info("No spatial data available for this filter combo.")
else:
    max_count = max(map_data["incident_count"].max(), 1)
    map_data["risk_level"] = map_data["incident_count"] / max_count

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=33.75,      
                longitude=-84.39,
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
            tooltip={"text": "NPU {npu}\nIncidents: {incident_count}"}
        )
    )
    st.caption(
        "Each circle represents an NPU. Larger circles = more incidents, based on current filters."
    )