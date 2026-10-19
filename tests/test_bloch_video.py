"""Bloch-sphere video of single-qubit runs, rendered with nutate."""

import sys
import warnings
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
    offsets = traj.offsets + p.Delta[0] / (2 * np.pi)  # traj is relative to Delta
    ref = nt.simulate(traj.sequence, offsets, m0=axis)
    np.testing.assert_allclose(traj.m, ref.m, atol=1e-10)


def test_bloch_trajectory_sweeps_the_optimised_band_in_hz():
    """Only Delta ± sw/2, the offsets the pulse was optimised for.

    The offsets are relative to Delta: nutate's colour bar is symmetric about
    zero, so absolute offsets (Delta = 10 MHz) would use only half of it.
    """
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory

    api = load_single_qubit_config()
    p = api.parameters
    delta, sw = api.config["parameters"]["Delta"][0], api.config["parameters"]["sw"][0]
    assert (delta, sw) == (10e6, 5e6)
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=11)

    np.testing.assert_allclose(
        traj.offsets, np.linspace(-sw / 2, sw / 2, 11), atol=1e-6
    )
    ((name, _, _),) = traj.segments
    # Plain text: nutate's video font has no Greek glyphs
    assert (
        name == traj.sequence[0].name == "optimised pulse, offsets relative to 10 MHz"
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


def test_render_bloch_video_writes_mp4(tmp_path):
    pytest.importorskip("nutate")
    from ctrl_freeq.visualisation.plotter import bloch_trajectory, render_bloch_video

    p = _single_qubit()
    traj = bloch_trajectory(_random_solution(p), p, p.init[0], num_points=5)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # e.g. "frames repeat"
        path = render_bloch_video(traj, tmp_path / "bloch.mp4")

    assert path == tmp_path / "bloch.mp4"
    assert path.read_bytes()[4:8] == b"ftyp"


def test_process_and_plot_gives_one_bloch_trajectory_per_initial_state():
    nt = pytest.importorskip("nutate")

    p = _single_qubit()
    _, figures = process_and_plot(_random_solution(p), p)

    bloch = [f for f in figures if isinstance(f, nt.Trajectory)]
    assert len(bloch) == len(p.init)
    assert not any(hasattr(f, "to_plotly_json") for f in figures)


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
