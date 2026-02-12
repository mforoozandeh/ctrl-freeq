# Dashboard

ctrl-freeq includes an interactive dashboard that combines all analysis figures from optimization runs into a single, shareable HTML file. It uses [Panel](https://panel.holoviz.org/) to create a responsive card grid with dark theme that works with both Matplotlib and Plotly figures.

---

## Features

- **No Conversion Required** — Accepts both Matplotlib and Plotly figures directly
- **Single-File Export** — Everything exported to one standalone HTML file
- **Dark Theme** — Clean, professional dark theme for better viewing
- **Interactive** — Plotly figures remain fully interactive
- **Responsive Layout** — Two-column card grid that adapts to screen size
- **Easy Sharing** — HTML file can be opened in any browser, no server needed

---

## Usage

### Automatic Generation (Recommended)

When you use the GUI's **Save Results** button, a dashboard is automatically generated alongside your plots in `results/dashboards/`.

### Programmatic Usage

```python
from ctrl_freeq.api import load_single_qubit_config
from ctrl_freeq.visualisation.plotter import process_and_plot
from ctrl_freeq.visualisation.dashboard import create_dashboard
from datetime import datetime

# Run optimization
api = load_single_qubit_config()
solution = api.run_optimization()

# Generate plots and get figures
waveforms, figures = process_and_plot(solution, api.parameters, save_plots=False)

# Create dashboard (pass parameters for a richer sidebar with algorithm, fidelity, etc.)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dashboard_path = create_dashboard(figures, timestamp, parameters=api.parameters)
print(f"Dashboard saved to: {dashboard_path}")
```

### `create_dashboard` Parameters

```python
create_dashboard(figures, timestamp, parameters=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `figures` | `list` | List of Matplotlib and/or Plotly figure objects |
| `timestamp` | `str` | Timestamp string for the dashboard filename |
| `parameters` | `Initialise`, optional | Parameters object from the optimization run. When provided, the dashboard sidebar displays algorithm, fidelity target, pulse parameters, and state information. |

!!! tip "Pass `parameters` for richer dashboards"
    Without the `parameters` argument, the dashboard sidebar only shows the timestamp and figure count. Passing `api.parameters` adds a detailed sidebar with all optimization and system parameters.

### Demo Function

Run the dashboard module directly to create a sample dashboard:

```bash
python -m ctrl_freeq.visualisation.dashboard
```

---

## Example Dashboard

Below is an example of the dashboard generated from a single-qubit optimization run. This is what you get when you click **Save Results** in the GUI or call `create_dashboard` via the API.

[View Example Dashboard](../assets/example_dashboard.html){ .md-button .md-button--primary target="_blank" }

The dashboard opens in a new tab as a standalone interactive page with Plotly figures you can zoom, pan, and hover over.

---

## Dashboard Contents

The dashboard typically includes:

| Category | Visualizations |
|----------|----------------|
| **Pulse Waveforms** | IQ plots, amplitude/phase evolution |
| **State Evolution** | History plots, observable dynamics |
| **Quantum State** | Bloch sphere (single qubit), population dynamics |
| **Analysis** | Excitation profiles, fidelity convergence |

---

## File Structure

```
ctrl-freeq/
├── src/
│   └── ctrl_freeq/
│       └── visualisation/
│           ├── __init__.py
│           ├── dashboard.py          # Dashboard creation module
│           ├── plotter.py            # Plotting functions
│           └── plot_settings.py      # Plot configuration
└── results/
    └── dashboards/                   # Generated dashboard HTML files
        └── dashboard_YYYYMMDD_HHMMSS.html
```

---

## Troubleshooting

!!! tip "Dashboard not generating?"
    Verify figures are being captured:
    ```python
    waveforms, figures = process_and_plot(solution, parameters, save_plots=False)
    print(f"Captured {len(figures)} figures")
    ```

!!! tip "Dashboard file too large?"
    The HTML file includes all resources inline for portability. To reduce size, use fewer figures or reduce figure resolution.

!!! tip "Figures not displaying correctly?"
    - Matplotlib figures: Ensure `tight=True` parameter is used
    - Plotly figures: Check that `plotly` extension is loaded with `pn.extension('plotly')`

---

## Next Steps

- [GUI Guide](gui.md) — Configure and run optimization interactively
- [API Reference](api.md) — Use ctrl-freeq programmatically
- [Panel Documentation](https://panel.holoviz.org/) — Learn more about Panel dashboards
