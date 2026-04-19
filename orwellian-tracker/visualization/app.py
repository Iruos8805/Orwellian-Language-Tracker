from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def filter_frame(df: pd.DataFrame, decades, party, chamber):
    out = df.copy()
    if "decade" in out.columns and decades:
        out = out[out["decade"].isin(decades)]
    if party != "All" and "party" in out.columns:
        out = out[out["party"] == party]
    if chamber != "All" and "chamber" in out.columns:
        out = out[out["chamber"] == chamber]
    return out


def main() -> None:
    st.set_page_config(page_title="Orwellian Tracker", layout="wide")
    st.title("Orwellian Language Decay Tracker")

    lexical = load_csv(BASE / "student1_lexical/diversity_scores.csv")
    syntactic = load_csv(BASE / "student2_syntactic/depth_scores.csv")
    semantic = load_csv(BASE / "student3_semantic/drift_rate_by_decade.csv")
    index = load_csv(BASE / "integration/newspeak_index.csv")
    by_party = load_csv(BASE / "integration/index_by_party.csv")
    by_chamber = load_csv(BASE / "integration/index_by_chamber.csv")

    if index.empty:
        st.warning("Run the pipeline first to populate integration/newspeak_index.csv")
        return

    decades_all = sorted(index["decade"].dropna().unique().tolist())
    decade_sel = st.sidebar.multiselect("Decades", decades_all, default=decades_all)
    party_options = (
        ["All"] + sorted(by_party["party"].dropna().unique().tolist())
        if not by_party.empty
        else ["All"]
    )
    chamber_options = (
        ["All"] + sorted(by_chamber["chamber"].dropna().unique().tolist())
        if not by_chamber.empty
        else ["All"]
    )
    party_sel = st.sidebar.selectbox("Party", party_options)
    chamber_sel = st.sidebar.selectbox("Chamber", chamber_options)

    index_f = filter_frame(index, decade_sel, party_sel, chamber_sel)
    lexical_f = filter_frame(lexical, decade_sel, party_sel, chamber_sel)
    syntactic_f = filter_frame(syntactic, decade_sel, party_sel, chamber_sel)
    semantic_f = filter_frame(semantic, decade_sel, party_sel, chamber_sel)

    if index_f.empty:
        st.info("No data for selected filters.")
        return

    latest = index_f.sort_values("decade").iloc[-1]
    previous = index_f.sort_values("decade").iloc[-2] if len(index_f) > 1 else latest
    delta = latest["Newspeak_Index"] - previous["Newspeak_Index"]
    st.metric("Newspeak Index", f"{latest['Newspeak_Index']:.4f}", f"{delta:+.4f}")

    c1, c2 = st.columns(2)
    with c1:
        fig_hero = px.line(
            lexical_f, x="decade", y="MATTR", markers=True, title="Hero graph: MATTR"
        )
        st.plotly_chart(fig_hero, use_container_width=True)
    with c2:
        fig_idx = px.line(
            index_f, x="decade", y="Newspeak_Index", markers=True, title="Unified index"
        )
        st.plotly_chart(fig_idx, use_container_width=True)

    fig_panel = px.line(
        pd.concat(
            [
                lexical_f[["decade", "Diversity_loss"]]
                .rename(columns={"Diversity_loss": "value"})
                .assign(metric="Diversity_loss"),
                syntactic_f[["decade", "Logic_flat"]]
                .rename(columns={"Logic_flat": "value"})
                .assign(metric="Logic_flat"),
                semantic_f[["decade", "Drift_rate"]]
                .rename(columns={"Drift_rate": "value"})
                .assign(metric="Drift_rate"),
            ],
            ignore_index=True,
        ),
        x="decade",
        y="value",
        color="metric",
        markers=True,
        title="Three-pillar trend panel",
    )
    st.plotly_chart(fig_panel, use_container_width=True)

    if not by_party.empty:
        party_f = filter_frame(by_party, decade_sel, party_sel, chamber_sel)
        heat = party_f.pivot(index="party", columns="decade", values="Newspeak_Index")
        st.subheader("Newspeak heatmap by party")
        st.dataframe(heat)

    if not by_chamber.empty:
        chamber_f = filter_frame(by_chamber, decade_sel, party_sel, chamber_sel)
        heat2 = chamber_f.pivot(
            index="chamber", columns="decade", values="Newspeak_Index"
        )
        st.subheader("Newspeak heatmap by chamber")
        st.dataframe(heat2)

    export_df = index_f.copy()
    st.download_button(
        label="Export filtered decade scores (CSV)",
        data=export_df.to_csv(index=False),
        file_name="filtered_decade_scores.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
