import streamlit as st

import common as C
from common import need_data, show, tip

DESC = {
    "Composition": "How the whole is split into parts — shares and proportions (pie, stacked bar, treemap).",
    "Distribution": "How values of a single variable are spread — centre, spread, skew, outliers (histogram, box, violin).",
    "Comparison": "How a measure differs across groups or over time (grouped bars, line charts).",
    "Relationship": "How variables move together (scatter, scatter matrix, correlation).",
}


def render():
    df = need_data()
    st.title("📊 Four Dimensions of Data Analysis")
    st.caption("Charts are picked automatically from the column types. Hover, zoom and click legend items to interact.")
    dims = C.c_dims(df)
    tabs = st.tabs(list(DESC))
    for tab, (name, figs) in zip(tabs, dims.items()):
        with tab:
            st.info(DESC[name])
            tip(name)
            if not figs:
                st.write("No suitable columns for this dimension.")
            for i, f in enumerate(figs):
                show(f, key=f"dim_{name}_{i}")
