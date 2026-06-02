from __future__ import annotations

import json
from pathlib import Path
import os

def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"')
        if key and val and key not in os.environ:
            os.environ[key] = val

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
from flask import Flask, jsonify, render_template

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent))

DATA_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_file(*relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = DATA_ROOT / relative_path
        if candidate.exists():
            return candidate
    options = ", ".join(str(DATA_ROOT / relative_path) for relative_path in relative_paths)
    raise FileNotFoundError(f"Expected one of these data files: {options}")


def load_plot_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    def _env_or_resolve(env_name: str, *candidates: str) -> Path:
        v = os.environ.get(env_name)
        if v:
            p = Path(v)
            if not p.is_absolute():
                p = DATA_ROOT / p
            if p.exists():
                return p
            raise FileNotFoundError(f"Environment {env_name} set to {v}, but file not found")
        if candidates:
            return _resolve_data_file(*candidates)
        raise FileNotFoundError(
            f"Required environment variable {env_name} is not set; set it to the path of the CSV input (absolute or relative to project root)."
        )
    lexical = pd.read_csv(_env_or_resolve("LEXICAL"))
    syntactic = pd.read_csv(_env_or_resolve("SYNTACTIC"))
    semantic = pd.read_csv(_env_or_resolve("SEMANTIC"))
    newspeak_index = pd.read_csv(_env_or_resolve("INDEX"))
    party = pd.read_csv(_env_or_resolve("PARTY"))
    semantic_words = pd.read_csv(_env_or_resolve("SEMANTIC_WORDS"))

    return lexical, syntactic, semantic, newspeak_index, party, semantic_words



PLOT_TEMPLATE = "plotly_dark"
PLOT_PAPER_BG = "#1a1a2e"
PLOT_BG = "#16213e"
PLOT_FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color="#ffffff")
PLOT_TITLE_FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=22, color="#e8c547")
PLOT_LEGEND = dict(
    bgcolor="rgba(22,33,62,0.8)",
    bordercolor="#92cc41",
    borderwidth=1,
    font=dict(size=12, color="#ffffff"),
)
PLOT_MARGIN = dict(l=70, r=30, t=80, b=70)


def fig_to_payload(fig: go.Figure) -> dict:
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


def style_figure(
    fig: go.Figure,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    height: int = 420,
) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=PLOT_PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=PLOT_FONT,
        legend=PLOT_LEGEND,
        margin=PLOT_MARGIN,
        title=dict(text=title, x=0.02, font=PLOT_TITLE_FONT),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text=xlabel,
        title_font=dict(size=14, color="#92cc41"),
        tickfont=dict(size=12, color="#92cc41"),
        gridcolor="#0f3460",
        zerolinecolor="#0f3460",
        linecolor="#92cc41",
    )
    fig.update_yaxes(
        title_text=ylabel,
        title_font=dict(size=14, color="#92cc41"),
        tickfont=dict(size=12, color="#92cc41"),
        gridcolor="#0f3460",
        zerolinecolor="#0f3460",
        linecolor="#92cc41",
    )
    return fig



def plot_hero_mattr(lex: pd.DataFrame) -> dict:
    decades = lex["decade"].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=decades,
            y=lex["MATTR"].to_numpy(),
            mode="lines+markers",
            name="MATTR",
            line=dict(color="#92cc41", width=3),
            marker=dict(size=10, color="#e8c547", line=dict(color="#ffffff", width=1)),
            fill="tozeroy",
            fillcolor="rgba(146,204,65,0.2)",
            hovertemplate="%{x}<br>MATTR: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0.50, 0.56])
    return fig_to_payload(style_figure(fig, "► LEXICAL DIVERSITY (MATTR) OVER TIME", xlabel="Decade", ylabel="MATTR Score"))


def plot_three_panel(lex: pd.DataFrame, syn: pd.DataFrame, sem: pd.DataFrame) -> dict:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=("LEXICAL", "SYNTACTIC", "SEMANTIC"),
    )

    fig.add_trace(
        go.Scatter(
            x=lex["decade"].astype(str).tolist(),
            y=lex["MATTR"].to_numpy(),
            mode="lines+markers",
            name="MATTR",
            line=dict(color="#92cc41", width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>MATTR: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lex["decade"].astype(str).tolist(),
            y=lex["Diversity_loss"].to_numpy(),
            mode="lines+markers",
            name="Diversity Loss",
            line=dict(color="#e8c547", width=3, dash="dash"),
            marker=dict(size=8, symbol="square"),
            hovertemplate="%{x}<br>Diversity loss: %{y:.3f}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=syn["decade"].astype(str).tolist(),
            y=syn["avg_amr_depth"].to_numpy(),
            mode="lines+markers",
            name="AMR Depth",
            line=dict(color="#92cc41", width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>AMR depth: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=syn["decade"].astype(str).tolist(),
            y=syn["Logic_flat"].to_numpy(),
            mode="lines+markers",
            name="Logic Flat",
            line=dict(color="#ff6b6b", width=3, dash="dash"),
            marker=dict(size=8, symbol="square"),
            hovertemplate="%{x}<br>Logic flat: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sem["decade"].astype(str).tolist(),
            y=sem["Drift_rate"].to_numpy(),
            mode="lines+markers",
            name="Drift Rate",
            line=dict(color="#ff6b6b", width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Drift rate: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02), height=1020)
    for row in (1, 2, 3):
        fig.update_xaxes(row=row, col=1, tickangle=45, tickfont=dict(size=12, color="#92cc41"), gridcolor="#0f3460", linecolor="#92cc41")
        fig.update_yaxes(row=row, col=1, tickfont=dict(size=12, color="#92cc41"), gridcolor="#0f3460", linecolor="#92cc41")
    fig.update_annotations(font=dict(size=16, color="#e8c547", family="Inter, Segoe UI, Arial, sans-serif"))
    return fig_to_payload(style_figure(fig, "", height=1020))


def plot_newspeak_heatmap(party: pd.DataFrame) -> dict:
    pivot = party.pivot(index="party", columns="decade", values="Newspeak_Index")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.astype(str).tolist(),
            y=pivot.index.astype(str).tolist(),
            colorscale="YlOrRd",
            zmin=float(np.nanmin(pivot.to_numpy())),
            zmax=float(np.nanmax(pivot.to_numpy())),
            text=pivot.round(3).astype(str),
            texttemplate="%{text}",
            textfont={"size": 12, "color": "#1a1a2e"},
            hovertemplate="Party: %{y}<br>Decade: %{x}<br>Index: %{z:.3f}<extra></extra>",
            colorbar=dict(title={"text": "Index", "font": {"size": 14, "color": "#92cc41"}}, tickfont=dict(size=12, color="#92cc41")),
        )
    )
    fig.update_layout(height=460, margin=dict(l=90, r=40, t=90, b=70))
    fig.update_xaxes(title_text="Decade", tickangle=45, tickfont=dict(size=12, color="#92cc41"), title_font=dict(size=14, color="#92cc41"))
    fig.update_yaxes(title_text="Party", tickfont=dict(size=12, color="#92cc41"), title_font=dict(size=14, color="#92cc41"))
    return fig_to_payload(style_figure(fig, "► NEWSPEAK INDEX HEATMAP BY PARTY", xlabel="Decade", ylabel="Party", height=460))


def plot_newspeak_line(idx: pd.DataFrame) -> dict:
    decades = idx["decade"].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=decades,
            y=idx["Newspeak_Index"].to_numpy(),
            mode="lines+markers",
            name="Newspeak Index",
            line=dict(color="#ff6b6b", width=3),
            marker=dict(size=10, color="#e8c547", line=dict(color="#ffffff", width=1)),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.25)",
            hovertemplate="%{x}<br>Newspeak index: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[-1, 0])
    return fig_to_payload(style_figure(fig, "► UNIFIED NEWSPEAK INDEX BY DECADE", xlabel="Decade", ylabel="Newspeak Index"))


def plot_semantic_shift_bar(drift_words: pd.DataFrame) -> dict | None:
    transitions = [("1990s", "2000s"), ("2000s", "2010s")]
    drift_words["decade_from"] = drift_words["decade_from"].astype(str)
    drift_words["decade"] = drift_words["decade"].astype(str)
    mask = drift_words.apply(lambda r: (r["decade_from"], r["decade"]) in transitions, axis=1)
    window = drift_words[mask].copy()
    if window.empty:
        return None
    top = (
        window.groupby("word", as_index=False)["cosine_drift"]
        .sum()
        .sort_values("cosine_drift", ascending=False)
        .head(12)
    )
    fig = go.Figure(
        data=go.Bar(
            x=top["cosine_drift"].to_numpy(),
            y=top["word"].astype(str).tolist(),
            orientation="h",
            marker=dict(
                color=top["cosine_drift"].to_numpy(),
                colorscale="YlOrRd",
                line=dict(color="#92cc41", width=1),
            ),
            hovertemplate="Word: %{y}<br>Cumulative drift: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig_to_payload(style_figure(fig, "► TOP SEMANTIC SHIFTS: 1990s → 2010s", xlabel="Cumulative Drift", ylabel="Word", height=540))


def plot_semantic_trajectories(drift_words: pd.DataFrame) -> dict | None:
    transitions = [("1990s", "2000s"), ("2000s", "2010s")]
    drift_words["decade_from"] = drift_words["decade_from"].astype(str)
    drift_words["decade"] = drift_words["decade"].astype(str)
    mask = drift_words.apply(lambda r: (r["decade_from"], r["decade"]) in transitions, axis=1)
    window = drift_words[mask].copy()
    if window.empty:
        return None
    top = (
        window.groupby("word", as_index=False)["cosine_drift"]
        .sum()
        .sort_values("cosine_drift", ascending=False)
        .head(6)
    )
    top_words = top["word"].astype(str).tolist()
    trend = (
        window[window["word"].isin(top_words)]
        .groupby(["word", "decade"], as_index=False)["cosine_drift"].mean()
    )
    palette = ["#92cc41", "#e8c547", "#ff6b6b", "#52b0e8", "#d46eb3", "#f0a500"]
    fig = go.Figure()
    for i, word in enumerate(top_words):
        sub = trend[trend["word"] == word].sort_values("decade")
        fig.add_trace(
            go.Scatter(
                x=sub["decade"].astype(str).tolist(),
                y=sub["cosine_drift"].to_numpy(),
                mode="lines+markers",
                name=word,
                line=dict(color=palette[i % len(palette)], width=3),
                marker=dict(size=8),
                hovertemplate=f"Word: {word}<br>%{{x}}<br>Drift: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(legend_title_text="Word")
    return fig_to_payload(style_figure(fig, "► DRIFT TRAJECTORIES FOR TOP SHIFTING WORDS", xlabel="Decade", ylabel="Cosine Drift", height=520))


def plot_diversity_loss(lex: pd.DataFrame) -> dict:
    decades = lex["decade"].astype(str).tolist()
    diversity_loss = lex["Diversity_loss"].to_numpy()
    fig = go.Figure(
        data=go.Bar(
            x=decades,
            y=diversity_loss,
            marker=dict(
                color=["#92cc41" if v < 0.3 else "#e8c547" if v < 0.45 else "#ff6b6b" for v in diversity_loss],
                line=dict(color="#1a1a2e", width=1.2),
            ),
            hovertemplate="%{x}<br>Diversity loss: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 0.05])
    return fig_to_payload(style_figure(fig, "► VOCABULARY DIVERSITY LOSS OVER DECADES", xlabel="Decade", ylabel="Diversity Loss Index"))



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/plots")
def api_plots():
    lex, syn, sem, idx, party, drift = load_plot_inputs()

    plots = {
        "hero_mattr": plot_hero_mattr(lex),
        "three_panel": plot_three_panel(lex, syn, sem),
        "newspeak_heatmap": plot_newspeak_heatmap(party),
        "newspeak_line": plot_newspeak_line(idx),
        "semantic_bar": plot_semantic_shift_bar(drift),
        "semantic_trajectories": plot_semantic_trajectories(drift),
        "diversity_loss": plot_diversity_loss(lex),
    }

    stats = {
        "mattr_latest": round(float(lex["MATTR"].iloc[-1]), 3),
        "mattr_change": round(float(lex["MATTR"].iloc[-1] - lex["MATTR"].iloc[0]), 3),
        "newspeak_latest": round(float(idx["Newspeak_Index"].iloc[-1]), 3),
        "newspeak_change": round(float(idx["Newspeak_Index"].iloc[-1] - idx["Newspeak_Index"].iloc[0]), 3),
        "drift_latest": round(float(sem["Drift_rate"].iloc[-1]), 3),
        "amr_change": round(float(syn["avg_amr_depth"].iloc[-1] - syn["avg_amr_depth"].iloc[0]), 3),
    }

    return jsonify({"plots": plots, "stats": stats})


if __name__ == "__main__":
    app.run(debug=True, port=5000)