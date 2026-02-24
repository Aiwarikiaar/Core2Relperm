"""
Core2Relperm — Streamlit MVP
Forward simulation of 1D two-phase core flooding with interactive parameter controls.
"""

import sys, os

# Add the library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Core2Relperm", layout="wide")
st.title("Core2Relperm — 1D Two-Phase Flow Simulator")

# ---------------------------------------------------------------------------
# Lazy-load the numba-compiled library (cache so JIT only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Compiling Numba kernels (first run only)…")
def load_library():
    from scallib001.displacementmodel1D2P001 import DisplacementModel1D2P
    import scallib001.relpermlib001 as rlplib
    # Warm up the JIT by running a trivial evaluation
    m = rlplib.Rlp2PCorey(0.1, 0.1, 2.0, 2.0, 0.5, 1.0)
    m.calc(np.linspace(0, 1, 5))
    return rlplib, DisplacementModel1D2P

rlplib, DisplacementModel1D2P = load_library()

# ---------------------------------------------------------------------------
# Sidebar — Input Parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Core & Fluid Properties")

col_core1, col_core2 = st.sidebar.columns(2)
with col_core1:
    core_length = st.number_input("Core length [cm]", value=3.9, min_value=0.1, step=0.1, format="%.2f")
    porosity = st.number_input("Porosity [v/v]", value=0.18, min_value=0.01, max_value=0.99, step=0.01, format="%.3f")
    viscosity_w = st.number_input("Water viscosity [cP]", value=1.0, min_value=0.01, step=0.1, format="%.2f")
    density_w = st.number_input("Water density [kg/m3]", value=1000.0, min_value=1.0, step=10.0, format="%.1f")

with col_core2:
    core_area = st.number_input("Core area [cm2]", value=12.0, min_value=0.1, step=0.1, format="%.2f")
    permeability = st.number_input("Permeability [mD]", value=100.0, min_value=0.01, step=10.0, format="%.1f")
    viscosity_n = st.number_input("Oil viscosity [cP]", value=1.0, min_value=0.01, step=0.1, format="%.2f")
    density_n = st.number_input("Oil density [kg/m3]", value=1000.0, min_value=1.0, step=10.0, format="%.1f")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Relative Permeability Model
# ---------------------------------------------------------------------------
st.sidebar.header("Relative Permeability Model")

rlp_type = st.sidebar.selectbox("Model type", ["Corey", "LET"])

col_rlp1, col_rlp2 = st.sidebar.columns(2)
with col_rlp1:
    Swc = st.number_input("Swc (Sr water)", value=0.08, min_value=0.0, max_value=0.5, step=0.01, format="%.3f")
    Ke_w = st.number_input("Ke water", value=0.70, min_value=0.01, max_value=1.0, step=0.05, format="%.3f")
with col_rlp2:
    Sorw = st.number_input("Sorw (Sr oil)", value=0.14, min_value=0.0, max_value=0.5, step=0.01, format="%.3f")
    Ke_o = st.number_input("Ke oil", value=1.00, min_value=0.01, max_value=1.0, step=0.05, format="%.3f")

if rlp_type == "Corey":
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        Nw = st.number_input("Nw (water exp.)", value=2.5, min_value=0.1, max_value=10.0, step=0.1, format="%.2f")
    with col_c2:
        Now = st.number_input("Now (oil exp.)", value=3.0, min_value=0.1, max_value=10.0, step=0.1, format="%.2f")
else:
    st.sidebar.markdown("**Water phase (LET)**")
    col_l1, col_l2, col_l3 = st.sidebar.columns(3)
    with col_l1:
        Lw = st.number_input("Lw", value=2.5, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")
    with col_l2:
        Ew = st.number_input("Ew", value=1.0, min_value=0.0, max_value=50.0, step=0.5, format="%.2f")
    with col_l3:
        Tw = st.number_input("Tw", value=1.5, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")
    st.sidebar.markdown("**Oil phase (LET)**")
    col_l4, col_l5, col_l6 = st.sidebar.columns(3)
    with col_l4:
        Lo = st.number_input("Lo", value=3.0, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")
    with col_l5:
        Eo = st.number_input("Eo", value=1.0, min_value=0.0, max_value=50.0, step=0.5, format="%.2f")
    with col_l6:
        To = st.number_input("To", value=1.5, min_value=0.01, max_value=10.0, step=0.1, format="%.2f")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Capillary Pressure (simple Skjaeveland via tabular)
# ---------------------------------------------------------------------------
st.sidebar.header("Capillary Pressure")

pc_enabled = st.sidebar.checkbox("Include capillary pressure", value=True)

if pc_enabled:
    col_pc1, col_pc2 = st.sidebar.columns(2)
    with col_pc1:
        pc_aw = st.number_input("aw (water exp.)", value=0.30, min_value=0.01, max_value=5.0, step=0.05, format="%.2f")
        pc_cw = st.number_input("cw [bar]", value=0.011, min_value=0.0001, max_value=1.0, step=0.001, format="%.4f")
    with col_pc2:
        pc_ao = st.number_input("ao (oil exp.)", value=0.90, min_value=0.01, max_value=5.0, step=0.05, format="%.2f")
        pc_Swi = st.number_input("Swi (imbibition)", value=0.13, min_value=0.0, max_value=0.99, step=0.01, format="%.3f")

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Injection Schedule
# ---------------------------------------------------------------------------
st.sidebar.header("Injection Schedule")

time_end = st.sidebar.number_input("Simulation end time [hours]", value=2.0, min_value=0.1, step=0.5, format="%.1f")

num_periods = st.sidebar.number_input("Number of injection periods", value=1, min_value=1, max_value=10, step=1)

schedule_rows = []
for i in range(int(num_periods)):
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        t_start = st.number_input(f"Start time {i+1} [hr]", value=0.0 if i == 0 else float(i), min_value=0.0, step=0.1, format="%.2f", key=f"sched_t_{i}")
    with col_s2:
        inj_rate = st.number_input(f"Rate {i+1} [cm3/min]", value=0.1 * (2**i), min_value=0.001, step=0.01, format="%.3f", key=f"sched_r_{i}")
    schedule_rows.append([t_start, inj_rate, 1.0])

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Simulation Settings
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")
NX = st.sidebar.slider("Grid cells (NX)", min_value=20, max_value=200, value=50, step=10)
refine_grid = st.sidebar.checkbox("Refine grid near boundaries", value=True)
gravity_on = st.sidebar.checkbox("Include gravity", value=False)

# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------
def build_and_run():
    """Construct models from sidebar parameters and run the solver."""
    # Relperm model
    if rlp_type == "Corey":
        rlp_model = rlplib.Rlp2PCorey(Swc, Sorw, Nw, Now, Ke_w, Ke_o)
    else:
        rlp_model = rlplib.Rlp2PLET(Swc, Sorw, Lw, Ew, Tw, Lo, Eo, To, Ke_w, Ke_o)

    # Capillary pressure model
    EPS_PC = 0.001
    Sorw_pc = max(Sorw - 0.01, 0.001)
    n_pts = 101

    if pc_enabled:
        # Build Skjaeveland Pc table, then interpolate with cubic spline
        swc_pc = Swc
        ssw_ref = (pc_Swi - swc_pc) / (1 - swc_pc)
        sso_ref = (1 - pc_Swi - Sorw_pc) / (1 - Sorw_pc)
        co = -pc_cw * np.power(sso_ref, pc_ao) / np.power(ssw_ref, pc_aw)

        sw_arr = np.linspace(swc_pc + EPS_PC, 1 - Sorw_pc - EPS_PC, n_pts)
        ssw = (sw_arr - swc_pc) / (1 - swc_pc)
        sso = (1 - sw_arr - Sorw_pc) / (1 - Sorw_pc)
        pc_arr = pc_cw / np.power(ssw, pc_aw) + co / np.power(sso, pc_ao)
        cpr_model = rlplib.CubicInterpolator(sw_arr, pc_arr, lex=1, rex=1)
    else:
        # Zero capillary pressure
        sw_arr = np.linspace(Swc + EPS_PC, 1 - Sorw_pc - EPS_PC, n_pts)
        pc_arr = np.zeros(n_pts)
        cpr_model = rlplib.CubicInterpolator(sw_arr, pc_arr, lex=0, rex=0)

    # Schedule
    schedule_df = pd.DataFrame(schedule_rows, columns=["StartTime", "InjRate", "FracFlow"])

    movie_times = np.linspace(0, time_end, 100)

    grav_mult = 1.0 if gravity_on else 0.0

    model = DisplacementModel1D2P(
        NX=NX,
        core_length=core_length,
        core_area=core_area,
        permeability=permeability,
        porosity=porosity,
        sw_initial=Swc,
        viscosity_w=viscosity_w,
        viscosity_n=viscosity_n,
        density_w=density_w,
        density_n=density_n,
        gravity_multiplier=grav_mult,
        rlp_model=rlp_model,
        cpr_model=cpr_model,
        time_end=time_end,
        rate_schedule=schedule_df,
        movie_schedule=movie_times,
        refine_grid=refine_grid,
    )

    model.solve()
    return model, rlp_model, cpr_model, sw_arr, pc_arr


# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running 1D2P solver…"):
        try:
            model, rlp_model, cpr_model, sw_pc_arr, pc_vals = build_and_run()
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

    st.success("Simulation complete!")

    res = model.results
    tss = res.tss_table

    # Store in session state so plots survive reruns
    st.session_state["results"] = res
    st.session_state["tss"] = tss
    st.session_state["model"] = model
    st.session_state["rlp_model"] = rlp_model
    st.session_state["sw_pc_arr"] = sw_pc_arr
    st.session_state["pc_vals"] = pc_vals

# ---------------------------------------------------------------------------
# Plots (shown if results exist in session)
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    res = st.session_state["results"]
    tss = st.session_state["tss"]
    model = st.session_state["model"]
    rlp_model_cached = st.session_state["rlp_model"]
    sw_pc_arr = st.session_state["sw_pc_arr"]
    pc_vals = st.session_state["pc_vals"]

    # ---- Tab layout -------------------------------------------------------
    tab_kr, tab_sim, tab_sat, tab_grid, tab_data = st.tabs(
        ["Rel-Perm & Pc", "Production & Pressure", "Saturation Profiles", "Simulation Grid", "Data Table"]
    )

    # ================================================================
    # TAB 1 — Relative Permeability & Capillary Pressure curves
    # ================================================================
    with tab_kr:
        swv = np.linspace(0, 1, 201)
        kr1 = rlp_model_cached.calc_kr1(swv)
        kr2 = rlp_model_cached.calc_kr2(swv)

        fig_kr = make_subplots(rows=1, cols=2, subplot_titles=("Relative Permeability", "Capillary Pressure"))

        fig_kr.add_trace(go.Scatter(x=swv, y=kr1, name="krw", line=dict(color="blue")), row=1, col=1)
        fig_kr.add_trace(go.Scatter(x=swv, y=kr2, name="kro", line=dict(color="red")), row=1, col=1)
        fig_kr.update_xaxes(title_text="Sw", row=1, col=1)
        fig_kr.update_yaxes(title_text="kr", range=[0, 1.05], row=1, col=1)

        fig_kr.add_trace(go.Scatter(x=sw_pc_arr, y=pc_vals, name="Pc", line=dict(color="green")), row=1, col=2)
        fig_kr.update_xaxes(title_text="Sw", row=1, col=2)
        fig_kr.update_yaxes(title_text="Pc [bar]", row=1, col=2)

        fig_kr.update_layout(height=420, margin=dict(t=40, b=40))
        st.plotly_chart(fig_kr, use_container_width=True)

    # ================================================================
    # TAB 2 — Production curves & pressure drop
    # ================================================================
    with tab_sim:
        fig_sim = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Pressure Drop vs Time",
                "Pressure Drop vs PV Injected",
                "Cumulative Oil Production",
                "Fractional Flow at Outlet",
            ),
        )

        # Pressure drop vs time
        fig_sim.add_trace(go.Scatter(x=tss["TIME"], y=tss["delta_P"], name="dP total", line=dict(color="black")), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=tss["TIME"], y=tss["delta_P_w"], name="dP water", line=dict(color="blue", dash="dash")), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=tss["TIME"], y=tss["delta_P_o"], name="dP oil", line=dict(color="red", dash="dash")), row=1, col=1)
        fig_sim.update_xaxes(title_text="Time [hr]", row=1, col=1)
        fig_sim.update_yaxes(title_text="dP [bar]", row=1, col=1)

        # Pressure drop vs PV injected
        fig_sim.add_trace(go.Scatter(x=tss["PVinj"], y=tss["delta_P"], name="dP total", showlegend=False, line=dict(color="black")), row=1, col=2)
        fig_sim.add_trace(go.Scatter(x=tss["PVinj"], y=tss["delta_P_w"], name="dP water", showlegend=False, line=dict(color="blue", dash="dash")), row=1, col=2)
        fig_sim.add_trace(go.Scatter(x=tss["PVinj"], y=tss["delta_P_o"], name="dP oil", showlegend=False, line=dict(color="red", dash="dash")), row=1, col=2)
        fig_sim.update_xaxes(title_text="PV Injected", row=1, col=2)
        fig_sim.update_yaxes(title_text="dP [bar]", row=1, col=2)

        # Cumulative oil production
        pore_volume = core_length * core_area * porosity
        fig_sim.add_trace(go.Scatter(x=tss["PVinj"], y=tss["CumOIL"] / pore_volume, name="Cum Oil / PV", line=dict(color="green")), row=2, col=1)
        fig_sim.update_xaxes(title_text="PV Injected", row=2, col=1)
        fig_sim.update_yaxes(title_text="Cum Oil [PV]", row=2, col=1)

        # Fractional flow at outlet
        fig_sim.add_trace(go.Scatter(x=tss["PVinj"], y=tss["FracFlowPrd"], name="Fw outlet", line=dict(color="purple")), row=2, col=2)
        fig_sim.update_xaxes(title_text="PV Injected", row=2, col=2)
        fig_sim.update_yaxes(title_text="Fw", range=[0, 1.05], row=2, col=2)

        fig_sim.update_layout(height=700, margin=dict(t=40, b=40))
        st.plotly_chart(fig_sim, use_container_width=True)

    # ================================================================
    # TAB 3 — Saturation profiles at selected times
    # ================================================================
    with tab_sat:
        movie_sw = res.movie_sw
        movie_tD = res.movie_tD
        xD = res.xD

        n_profiles = len(movie_tD)
        # Pick ~10 evenly-spaced profiles
        indices = np.linspace(0, n_profiles - 1, min(10, n_profiles), dtype=int)

        fig_sat = go.Figure()
        for idx in indices:
            fig_sat.add_trace(
                go.Scatter(
                    x=xD,
                    y=movie_sw[idx],
                    name=f"PVinj={movie_tD[idx]:.3f}",
                    mode="lines",
                )
            )

        fig_sat.update_layout(
            title="Water Saturation Profiles Along Core",
            xaxis_title="Position x/L",
            yaxis_title="Sw",
            yaxis_range=[0, 1],
            height=450,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_sat, use_container_width=True)

    # ================================================================
    # TAB 4 — Simulation Grid Visualization
    # ================================================================
    with tab_grid:
        movie_sw_g = res.movie_sw
        movie_tD_g = res.movie_tD
        xD_g = res.xD
        x_g = res.x

        # --- Sub-plot 1: Space-time heatmap ---
        st.subheader("Space-Time Saturation Map")
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                x=x_g,
                y=movie_tD_g,
                z=movie_sw_g,
                colorscale="RdYlBu",
                colorbar=dict(title="Sw"),
                zmin=0,
                zmax=1,
            )
        )
        fig_heatmap.update_layout(
            xaxis_title="Position along core [cm]",
            yaxis_title="PV Injected",
            height=420,
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # --- Sub-plot 2: Grid cell diagram ---
        st.subheader("Grid Cell Sizes")
        cell_widths = np.diff(np.concatenate([[0], 0.5 * (x_g[:-1] + x_g[1:]), [core_length]]))
        cell_colors = cell_widths / cell_widths.max()

        fig_cells = go.Figure()
        for i in range(len(x_g)):
            fig_cells.add_trace(
                go.Bar(
                    x=[x_g[i]],
                    y=[cell_widths[i]],
                    width=[cell_widths[i] * 0.95],
                    marker_color=f"rgba(31,119,180,{0.3 + 0.7 * cell_colors[i]:.2f})",
                    hovertext=f"Cell {i+1}<br>Centre: {x_g[i]:.4f} cm<br>Width: {cell_widths[i]:.4f} cm",
                    hoverinfo="text",
                    showlegend=False,
                )
            )
        fig_cells.update_layout(
            xaxis_title="Position along core [cm]",
            yaxis_title="Cell width [cm]",
            height=250,
            margin=dict(t=10, b=40),
            bargap=0,
        )
        st.plotly_chart(fig_cells, use_container_width=True)

        st.caption(
            f"Grid: **{len(x_g)}** cells | "
            f"Min width: **{cell_widths.min():.4f}** cm | "
            f"Max width: **{cell_widths.max():.4f}** cm | "
            f"Refinement: **{'On' if refine_grid else 'Off'}**"
        )

        # --- Sub-plot 3: Interactive 1D saturation strip at selected time ---
        st.subheader("Saturation at Selected Time")
        n_steps = len(movie_tD_g)
        step_idx = st.slider(
            "Select timestep",
            min_value=0,
            max_value=n_steps - 1,
            value=n_steps - 1,
            format=f"Step %d",
            key="grid_slider",
        )
        st.write(f"PV Injected: **{movie_tD_g[step_idx]:.4f}**")

        sw_snap = movie_sw_g[step_idx]

        fig_strip = go.Figure()
        # Colored bar for each cell
        for i in range(len(x_g)):
            fig_strip.add_shape(
                type="rect",
                x0=x_g[i] - cell_widths[i] / 2,
                x1=x_g[i] + cell_widths[i] / 2,
                y0=0,
                y1=1,
                fillcolor=f"rgba({int(255*(1-sw_snap[i]))}, {int(100+100*sw_snap[i])}, {int(255*sw_snap[i])}, 0.85)",
                line_width=0.5,
                line_color="white",
            )
        # Invisible scatter for hover info
        fig_strip.add_trace(
            go.Bar(
                x=x_g,
                y=[1] * len(x_g),
                width=cell_widths,
                marker_color=[
                    f"rgba({int(255*(1-s))}, {int(100+100*s)}, {int(255*s)}, 0.85)"
                    for s in sw_snap
                ],
                customdata=np.column_stack([np.arange(1, len(x_g) + 1), sw_snap, cell_widths]),
                hovertemplate="Cell %{customdata[0]:.0f}<br>Sw: %{customdata[1]:.4f}<br>Width: %{customdata[2]:.4f} cm<extra></extra>",
                showlegend=False,
            )
        )
        fig_strip.update_layout(
            xaxis_title="Position along core [cm]",
            yaxis=dict(visible=False, range=[0, 1]),
            height=120,
            margin=dict(t=5, b=40, l=40, r=40),
            bargap=0,
        )
        st.plotly_chart(fig_strip, use_container_width=True)

    # ================================================================
    # TAB 5 — Raw data table
    # ================================================================
    with tab_data:
        st.dataframe(tss, use_container_width=True, height=500)
        csv = tss.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="core2relperm_results.csv", mime="text/csv")

else:
    st.info("Adjust parameters in the sidebar, then click **Run Simulation**.")
