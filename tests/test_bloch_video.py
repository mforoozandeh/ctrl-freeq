"""Bloch-sphere video of single-qubit runs, rendered with nutate."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ctrl_freeq.api import (
    load_single_qubit_config,
    load_single_qubit_dissipative_config,
)
from ctrl_freeq.visualisation.plotter import (
    _plotter_waveforms,
    process_and_plot,
)


def _random_solution(p, scale=0.5, seed=0):
    torch.manual_seed(seed)
    return scale * torch.randn(sum(p.n_para_updated), dtype=torch.float64)


def _single_qubit(space="hilbert", initial="Z"):
    api = load_single_qubit_config()
    api.update_parameter("initial_states", [[initial]])
    if space != "hilbert":
        api.update_parameter("optimization.space", space)
    return api.parameters


@pytest.mark.parametrize("space", ["hilbert", "liouville"])
@pytest.mark.parametrize("initial", ["Z", "X", "-Y"])
def test_bloch_trajectory_matches_nutate_simulation(space, initial):
    """ctrl-freeq's replay and nutate's own Bloch simulation agree."""
    nt = pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    p = _single_qubit(space, initial)
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=11)

    axis = {"Z": [0, 0, 1], "X": [1, 0, 0], "-Y": [0, -1, 0]}[initial]
    np.testing.assert_allclose(traj.m[0], np.tile(axis, (11, 1)), atol=1e-12)
    ref = nt.simulate(traj.sequence, traj.offsets, m0=axis)
    np.testing.assert_allclose(traj.m, ref.m, atol=1e-10)


def test_bloch_trajectory_sweeps_the_optimised_band_in_hz():
    """Only Delta ± sw/2, the offsets the pulse was optimised for."""
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    api = load_single_qubit_config()
    p = api.parameters
    delta, sw = api.config["parameters"]["Delta"][0], api.config["parameters"]["sw"][0]
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=11)

    np.testing.assert_allclose(
        traj.offsets, np.linspace(delta - sw / 2, delta + sw / 2, 11), rtol=1e-12
    )


def test_excitation_profile_still_looks_beyond_the_optimised_band():
    """The profile plot keeps its 1.5 * sw sweep, in MHz."""
    from ctrl_freeq.visualisation.plotter import plot_excitation_profiles

    api = load_single_qubit_config()
    p = api.parameters
    delta, sw = api.config["parameters"]["Delta"][0], api.config["parameters"]["sw"][0]
    fig = plot_excitation_profiles(_random_solution(p), p, p.init[0], 11)

    f = fig.axes[0].lines[0].get_xdata()
    plt.close(fig)
    expected = np.linspace(delta - 0.75 * sw, delta + 0.75 * sw, 11) / 1e6
    np.testing.assert_allclose(f, expected, rtol=1e-12)


def test_bloch_trajectory_has_a_state_for_every_video_frame():
    """nutate's MP4 is 10 s at 30 fps; a 100-sample pulse is split into substeps."""
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    p = _single_qubit()
    assert p.np_pulse < 300
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=5)

    n_steps = traj.t.size - 1
    assert n_steps >= 300 and n_steps % p.np_pulse == 0
    k = n_steps // p.np_pulse
    np.testing.assert_allclose(traj.t[::k], p.state_boundary_times(), rtol=1e-12)


def test_bloch_trajectory_keeps_drive_above_nutate_amplitude_limit():
    """The amplitude limit is a soft penalty, so peaks above nutate's 1.5 occur."""
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    p = _single_qubit()
    x = _random_solution(p, scale=20.0)
    _amps, cx, cy = _plotter_waveforms(x, p)
    amplitude = torch.hypot(cx[:, 0], cy[:, 0]).numpy()
    assert amplitude.max() > 1.5

    traj = bloch_trajectory(x, p, p.init[0], num_points=5)

    (segment,) = traj.sequence
    k = segment.n_steps // p.np_pulse
    np.testing.assert_allclose(
        segment.drive,
        np.repeat(amplitude, k) * p.Omega_R_max[0] / (2 * np.pi),
        rtol=1e-12,
    )


def test_bloch_trajectory_shows_dissipative_decay():
    """No drive for T = T1 from -Z: z relaxes to 1 - 2/e at every offset."""
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    api = load_single_qubit_dissipative_config()
    api.update_parameter("initial_states", [["-Z"]])
    api.update_parameter("parameters.T2", [2e-7])  # before T1: T2 <= 2*T1
    api.update_parameter("parameters.T1", [2e-7])
    p = api.parameters
    x = torch.zeros(sum(p.n_para_updated), dtype=torch.float64)

    traj = bloch_trajectory(x, p, p.init[0], num_points=5)

    np.testing.assert_allclose(traj.m[-1, :, 2], 1 - 2 * np.exp(-1.0), atol=1e-6)
    np.testing.assert_allclose(traj.m[-1, :, :2], 0.0, atol=1e-12)


def test_render_bloch_video_writes_mp4(tmp_path, capfd):
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory, render_bloch_video

    p = _single_qubit()
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=5)

    path = render_bloch_video(traj, tmp_path / "bloch.mp4")

    assert path == tmp_path / "bloch.mp4"
    assert path.read_bytes()[4:8] == b"ftyp"
    assert "frames repeat" not in capfd.readouterr().err  # nutate's warning


@pytest.fixture
def crashing_python(tmp_path, monkeypatch):
    """Child processes die of SIGSEGV, as VTK does without a display."""
    script = tmp_path / "python"
    script.write_text("#!/bin/sh\nkill -SEGV $$\n")
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(script))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal")
def test_render_bloch_video_survives_a_crashing_renderer(
    tmp_path, crashing_python, capsys
):
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory, render_bloch_video

    p = _single_qubit()
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=5)

    assert render_bloch_video(traj, tmp_path / "bloch.mp4") is None
    assert not (tmp_path / "bloch.mp4").exists()
    assert "Bloch-sphere video skipped" in capsys.readouterr().out


def test_process_and_plot_gives_one_bloch_trajectory_per_initial_state():
    nt = pytest.importorskip("nutate")

    p = _single_qubit()
    _, figures = process_and_plot(_random_solution(p), p)

    bloch = [f for f in figures if isinstance(f, nt.Trajectory)]
    assert len(bloch) == len(p.init)
    assert all(isinstance(f, (plt.Figure, nt.Trajectory)) for f in figures)


def test_process_and_plot_skips_bloch_video_without_nutate(monkeypatch):
    monkeypatch.setitem(sys.modules, "nutate", None)

    p = _single_qubit()
    _, figures = process_and_plot(_random_solution(p), p)

    assert all(isinstance(f, plt.Figure) for f in figures)
    assert len(figures) == 5  # IQ, amplitude/phase, history, observables, profiles


@pytest.fixture
def dashboard_files():
    """Remove what create_dashboard writes into results/dashboards."""
    written = []
    yield written
    for path in written:
        Path(path).unlink(missing_ok=True)


def _tiny_render(traj, path):
    import nutate as nt

    return nt.render(traj, path, duration=0.2, hold=0, size=(160, 96))


def test_dashboard_embeds_bloch_video_file(tmp_path, dashboard_files):
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.dashboard import create_dashboard
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    p = _single_qubit()
    x = _random_solution(p)
    video = _tiny_render(
        bloch_trajectory(x, p, p.init[0], num_points=5), tmp_path / "b.mp4"
    )
    _, figures = process_and_plot(x, p)
    figures = [video if not isinstance(f, plt.Figure) else f for f in figures]

    html = create_dashboard(figures, "pytest_bloch_file", p)
    dashboard_files.append(html)

    text = Path(html).read_text()
    assert "data:video/mp4;base64," in text
    assert "Bloch Sphere" in text
    assert "Excitation Profiles (Frequency Response)" in text


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal")
def test_dashboard_skips_bloch_card_when_rendering_crashes(
    crashing_python, dashboard_files
):
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.dashboard import create_dashboard

    p = _single_qubit()
    _, figures = process_and_plot(_random_solution(p), p)

    html = create_dashboard(figures, "pytest_bloch_crash", p)
    dashboard_files.extend(
        [html, *Path(html).parent.glob("bloch_pytest_bloch_crash_*")]
    )

    text = Path(html).read_text()
    assert "data:video/mp4" not in text
    assert "Bloch Sphere" not in text
    assert "Excitation Profiles (Frequency Response)" in text


def test_dashboard_renders_bloch_trajectory(monkeypatch, dashboard_files):
    pytest.importorskip("nutate")
    import ctrl_freeq.visualisation.plotter as plotter
    from ctrl_freeq.visualisation.dashboard import create_dashboard

    monkeypatch.setattr(plotter, "render_bloch_video", _tiny_render)
    p = _single_qubit()
    _, figures = process_and_plot(_random_solution(p), p)

    html = create_dashboard(figures, "pytest_bloch_traj", p)
    videos = sorted(Path(html).parent.glob("bloch_pytest_bloch_traj_*.mp4"))
    dashboard_files.extend([html, *videos])

    assert len(videos) == 1
    assert "data:video/mp4;base64," in Path(html).read_text()


def test_dashboard_needs_no_plotly(monkeypatch, dashboard_files):
    """plotly is not a dependency: ctrl-freeq's own figures are matplotlib and video."""
    from ctrl_freeq.visualisation.dashboard import create_dashboard

    for name in [m for m in sys.modules if m == "plotly" or m.startswith("plotly.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "nutate", None)
    p = _single_qubit()
    _, figures = process_and_plot(_random_solution(p), p)

    html = create_dashboard(figures, "pytest_no_plotly", p)
    dashboard_files.append(html)

    text = Path(html).read_text()
    assert "Excitation Profiles (Frequency Response)" in text
    assert "plotlyplot/" not in text  # no plotly.js download


def test_dashboard_rejects_plotly_figures():
    """Plotly support is gone: say so instead of failing inside Panel."""
    from ctrl_freeq.visualisation.dashboard import create_dashboard

    class PlotlyLike:
        def to_plotly_json(self):
            return {"data": [], "layout": {}}

    with pytest.raises(TypeError, match="PlotlyLike"):
        create_dashboard([PlotlyLike()], "pytest_rejects_plotly")
