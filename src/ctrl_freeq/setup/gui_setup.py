import tkinter as tk
from tkinter import ttk
import json

import numpy as np
import torch

from ctrl_freeq.run.run_ctrl import run_ctrl
from ctrl_freeq.setup.initialise_gui import Initialise
from ctrl_freeq.visualisation.plotter import process_and_plot
from ctrl_freeq.optimizers.qiskit_optimizers import get_supported_qiskit_optimizers

sigma_J_entry = None


def validate_integer(P):
    if P.isdigit() or P == "":
        return True
    return False


def validate_float(P):
    try:
        float(P)
        return True
    except ValueError:
        if P == "" or "e" in P.lower():
            return True
        return False


def validate_string(P):
    return True  # Allow any input


def is_numeric(val):
    return isinstance(val, (int, float, complex))


def convert_to_array(data):
    """Recursively checks if all elements are numeric and converts to a NumPy array if they are."""
    if isinstance(data, list):
        # Check for matrix (list of lists)
        if all(isinstance(i, list) for i in data):
            if all(is_numeric(j) for i in data for j in i):
                return np.array(data)
            else:
                return data
        # Check for simple list
        elif all(is_numeric(i) for i in data):
            return np.array(data)
        else:
            return data
    else:
        return data


def split_string(input_list):
    result = []
    for elem in input_list:
        res = [item.strip() for item in elem.split(",")] if elem else None
        result.append(res)
    return result


def reshape_list(input_list):
    if all(elem is None for elem in input_list):
        return input_list
    m = len(input_list)
    n = len(input_list[0])
    new_list = []
    for i in range(n):
        new_list.append([input_list[j][i] for j in range(m)])
    return new_list


def add_coupling_entries():
    global coupling_entries, coupling_type_var, sigma_J_entry
    for widget in coupling_frame.winfo_children():
        widget.destroy()

    coupling_entries = []
    if qubit_count > 1:
        # Add coupling type dropdown
        ttk.Label(coupling_frame, text="Coupling Type:").grid(
            row=0, column=0, sticky="e"
        )
        coupling_type_var = tk.StringVar(value="XY")
        coupling_type_combobox = ttk.Combobox(
            coupling_frame, textvariable=coupling_type_var, values=["Z", "XYZ", "XY"]
        )
        coupling_type_combobox.grid(row=0, column=1, sticky="e")

        ttk.Label(coupling_frame, text="Coupling Constants (J_ij):").grid(
            row=1, column=0, columnspan=qubit_count, sticky="e"
        )
        row = 2

        for i in range(1, qubit_count + 1):
            for j in range(i + 1, qubit_count + 1):
                label = ttk.Label(coupling_frame, text=f"J_{i}{j}:")
                label.grid(row=row, column=0, sticky="e")
                entry = ttk.Entry(
                    coupling_frame, validate="key", validatecommand=vcmd_float
                )
                entry.grid(row=row, column=1, sticky="e")
                coupling_entries.append((i, j, entry))
                row += 1

        # Add sigma_J entry
        ttk.Label(coupling_frame, text="σ J:").grid(row=row, column=0, sticky="e")
        sigma_J_entry = ttk.Entry(
            coupling_frame, validate="key", validatecommand=vcmd_float
        )
        sigma_J_entry.grid(row=row, column=1, sticky="e")


def clean_data(data):
    if isinstance(data, dict):
        cleaned_dict = {}
        for k, v in data.items():
            cleaned_value = clean_data(v)
            # Only add to the dictionary if the cleaned value is not None and not an empty list or dict
            if cleaned_value not in [None, [], {}]:
                cleaned_dict[k] = cleaned_value
        return cleaned_dict
    elif isinstance(data, list):
        # Clean the list and remove any None or empty elements
        cleaned_list = [clean_data(v) for v in data if v not in [None, [], {}]]
        return cleaned_list
    else:
        return data


def make_data():
    data = {
        "qubits": [],
        "compute_resource": compute_resource_var.get(),
        "parameters": {
            "Delta": [],
            "sigma_Delta": [],
            "Omega_R_max": [],
            "pulse_duration": [],
            "point_in_pulse": [],
            "wf_type": [],
            "wf_mode": [],
            "amplitude_envelope": [],
            "amplitude_order": [],
            "coverage": [],
            "sw": [],
            "pulse_offset": [],
            "pulse_bandwidth": [],
            "ratio_factor": [],
            "sigma_Omega_R_max": [],
            "profile_order": [],
            "n_para": [],
            "J": [
                [0.0 for _ in range(qubit_count)] for _ in range(qubit_count)
            ],  # Initialize J as a matrix
            "coupling_type": coupling_type_var.get()
            if qubit_count > 1
            else None,  # Add coupling type
            "sigma_J": float(sigma_J_entry.get())
            if sigma_J_entry is not None and sigma_J_entry.get()
            else None,  # Add sigma_J
        },
        "initial_states": [init_ax_entry.get() for init_ax_entry in init_ax_entries],
        "target_states": {
            "Axis": [
                axis_entry.get() if target_method_var.get() == "Axis" else None
                for axis_entry in axis_entries
            ],
            "Phi": [
                phi_entry.get() if target_method_var.get() == "Phi and Beta" else None
                for phi_entry in phi_entries
            ],
            "Beta": [
                beta_entry.get() if target_method_var.get() == "Phi and Beta" else None
                for beta_entry in beta_entries
            ],
            "Gate": [
                gate_entry.get() if target_method_var.get() == "Gate" else None
                for gate_entry in gate_entries
            ],
        },
        "optimization": {
            "space": space_var.get(),
            "H0_snapshots": int(H0_snapshots_entry.get()),
            "Omega_R_snapshots": int(omega_1_snapshots_entry.get()),
            "algorithm": algorithm_var.get(),
            "max_iter": int(max_iter_entry.get()),
            "targ_fid": float(targ_fid_entry.get()),
        },
    }

    for i in range(len(qubits_vars)):
        data["qubits"].append(qubits_vars[i].get())
        data["parameters"]["Delta"].append(float(Omega_entries[i].get()))
        data["parameters"]["sigma_Delta"].append(float(sigma_Omega_entries[i].get()))
        data["parameters"]["Omega_R_max"].append(float(omega_1_max_entries[i].get()))
        data["parameters"]["pulse_duration"].append(
            float(pulse_duration_entries[i].get())
        )
        data["parameters"]["point_in_pulse"].append(
            int(point_in_pulse_entries[i].get())
        )
        data["parameters"]["wf_type"].append(wf_type_vars[i].get())
        data["parameters"]["wf_mode"].append(wf_mode_vars[i].get())
        data["parameters"]["amplitude_envelope"].append(
            amplitude_envelope_vars[i].get()
        )
        data["parameters"]["amplitude_order"].append(
            int(amplitude_order_entries[i].get())
        )
        data["parameters"]["coverage"].append(coverage_vars[i].get())
        data["parameters"]["sw"].append(float(sw_entries[i].get()))
        data["parameters"]["pulse_offset"].append(float(pulse_offset_entries[i].get()))
        data["parameters"]["pulse_bandwidth"].append(
            float(pulse_bandwidth_entries[i].get())
        )
        data["parameters"]["ratio_factor"].append(0.5)
        data["parameters"]["sigma_Omega_R_max"].append(
            float(sigma_omega_1_max_entries[i].get())
        )
        data["parameters"]["profile_order"].append(int(order_entries[i].get()))
        data["parameters"]["n_para"].append(int(n_para_entries[i].get()))

    for i, j, entry in coupling_entries:
        data["parameters"]["J"][i - 1][j - 1] = float(entry.get())

    data["initial_states"] = split_string(data["initial_states"])
    data["initial_states"] = reshape_list(data["initial_states"])

    data["target_states"]["Axis"] = split_string(data["target_states"]["Axis"])
    data["target_states"]["Axis"] = reshape_list(data["target_states"]["Axis"])

    data["target_states"]["Phi"] = split_string(data["target_states"]["Phi"])
    data["target_states"]["Phi"] = reshape_list(data["target_states"]["Phi"])

    data["target_states"]["Beta"] = split_string(data["target_states"]["Beta"])
    data["target_states"]["Beta"] = reshape_list(data["target_states"]["Beta"])

    data["target_states"]["Gate"] = split_string(data["target_states"]["Gate"])
    data["target_states"]["Gate"] = data["target_states"]["Gate"][0]

    data["target_states"]["Beta"] = [
        [float(item) for item in sublist] if sublist is not None else sublist
        for sublist in data["target_states"]["Beta"]
    ]

    return data


def save_to_json():
    import os

    data = make_data()
    data = clean_data(data)
    # Get the absolute path to the run directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(os.path.dirname(script_dir), "run")
    # Create the directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Save the file
    file_path = os.path.join(run_dir, "single_qubit_parameters_polar_phase.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Parameters saved to {file_path}")


def make_data_with_arrays():
    data = make_data()
    data = clean_data(data)
    # Convert applicable lists to NumPy arrays using convert_to_array
    data["parameters"] = {
        key: convert_to_array(value) for key, value in data["parameters"].items()
    }
    data["target_states"] = {
        key: convert_to_array(value) for key, value in data["target_states"].items()
    }
    return data


def verify_solution_file(filename, original_solution):
    """
    Verify that the saved solution file is correct by loading it and comparing with the original solution.

    Args:
        filename (str): Path to the saved solution file
        original_solution (numpy.ndarray): The original solution to compare with

    Returns:
        bool: True if the file is valid, False otherwise
    """
    import numpy as np

    try:
        # Load the saved solution
        loaded_solution = np.load(filename)

        # Check if the loaded solution has the same shape as the original
        if loaded_solution.shape != original_solution.shape:
            print(
                f"Warning: Loaded solution has different shape: {loaded_solution.shape} vs {original_solution.shape}"
            )
            return False

        # Check if the loaded solution is close to the original
        if not np.allclose(loaded_solution, original_solution):
            print("Warning: Loaded solution differs from original solution")
            return False

        print(f"Solution file verified successfully: {filename}")
        return True
    except Exception as e:
        print(f"Error verifying solution file: {e}")
        return False


def run_optimisation_and_plot():
    global last_solution, last_parameters, last_timestamp

    # Free memory from any previous run before allocating new objects
    last_solution = None
    last_parameters = None
    last_timestamp = None

    data = make_data_with_arrays()
    p = Initialise(data)
    sol = run_ctrl(p)
    sol = sol.detach().cpu().numpy()

    # Save the optimization solution to a file
    import numpy as np
    from datetime import datetime

    # Create a results directory if it doesn't exist
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / "results"
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = str(results_dir / f"solution_{timestamp}.npy")

    # Save the solution
    np.save(filename, sol)
    print(f"Solution saved to {filename}")

    # Verify the saved solution file
    verify_solution_file(filename, sol)

    # If sol came back as a NumPy array, turn it into a Torch tensor
    if isinstance(sol, np.ndarray):
        sol = torch.tensor(sol, dtype=torch.get_default_dtype())

    # Process and plot without saving plots initially
    waveforms, figures = process_and_plot(sol, p, save_plots=False)

    # Store the results for later use by the Save Results button
    last_solution = sol
    last_parameters = p
    last_timestamp = timestamp

    # Return the waveforms
    return waveforms


def save_results():
    """
    Save the results (plots and waveforms) from the last optimization run.
    This function is called when the "Save Results" button is clicked.
    """
    global last_solution, last_parameters, last_timestamp

    if last_solution is None or last_parameters is None or last_timestamp is None:
        print("No results to save. Please run the optimization first.")
        return

    import numpy as np

    # Create results directories if they don't exist
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / "results"
    plots_dir = results_dir / "plots"
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True)

    # Save the waveforms and get figures
    waveforms, figures = process_and_plot(
        last_solution, last_parameters, save_plots=True
    )
    waveform_filename = str(results_dir / f"waveforms_{last_timestamp}.npy")
    # Using allow_pickle=True because waveforms contains Python objects
    np.save(waveform_filename, waveforms, allow_pickle=True)
    print(f"Waveforms saved to {waveform_filename}")
    print("Note: When loading this file, use np.load(filename, allow_pickle=True)")

    # Generate dashboard
    from ctrl_freeq.visualisation.dashboard import create_dashboard

    dashboard_path = create_dashboard(figures, last_timestamp, last_parameters)
    print(f"Dashboard saved to {dashboard_path}")

    # Free the heavy Initialise object now that results are saved to disk
    last_parameters = None
    import gc
    gc.collect()

    print("Results saved successfully!")


def update_pulse_duration_entries(event):
    source_entry = event.widget
    new_value = source_entry.get()
    for entry in pulse_duration_entries:
        if entry != source_entry:
            entry.delete(0, tk.END)
            entry.insert(0, new_value)


def update_point_in_pulse_entries(event):
    source_entry = event.widget
    new_value = source_entry.get()
    for entry in point_in_pulse_entries:
        if entry != source_entry:
            entry.delete(0, tk.END)
            entry.insert(0, new_value)


def update_gate_entries(event):
    source_entry = event.widget
    new_value = source_entry.get()
    for entry in gate_entries:
        if entry != source_entry:
            entry.delete(0, tk.END)
            entry.insert(0, new_value)


def update_coverage_fields():
    """Update visibility of pulse_bandwidth and order fields based on coverage selection."""
    for i in range(qubit_count):
        coverage = coverage_vars[i].get()
        if coverage in ["selective", "band_selective"]:
            # Show pulse_bandwidth and order entries for this qubit
            pulse_bandwidth_entries[i].grid(row=15, column=i + 1)
            order_entries[i].grid(row=19, column=i + 1)
        else:
            # Hide pulse_bandwidth and order entries for this qubit
            pulse_bandwidth_entries[i].grid_remove()
            order_entries[i].grid_remove()

    # Show/hide labels based on whether any qubit has selective/band_selective coverage
    any_selective = any(
        coverage_vars[i].get() in ["selective", "band_selective"]
        for i in range(qubit_count)
    )
    if any_selective:
        pulse_bandwidth_label.grid(row=15, column=0)
        order_label.grid(row=19, column=0)
    else:
        pulse_bandwidth_label.grid_remove()
        order_label.grid_remove()


def update_target_fields():
    # Update the labels and visibility based on the selected target method
    method = target_method_var.get()

    if method == "Axis":
        axis_label.config(text="Axis:")
        beta_label.grid_remove()
        phi_label.grid_remove()
        gate_label.grid_remove()
        for i in range(qubit_count):
            axis_entries[i].grid(row=27, column=i + 1, padx=(10, 0), pady=2, sticky="w")
            if phi_entries[i]:
                phi_entries[i].grid_remove()
            if beta_entries[i]:
                beta_entries[i].grid_remove()
            if gate_entries[i]:
                gate_entries[i].grid_remove()

    elif method == "Gate":
        gate_label.config(text="Gate:")
        beta_label.grid_remove()
        phi_label.grid_remove()
        axis_label.grid_remove()
        for i in range(qubit_count):
            # Add gate entry and bind the update function
            if i >= len(gate_entries):  # Ensure only new entries are added
                gate_entry = ttk.Entry(
                    root, validate="key", validatecommand=vcmd_string
                )
                gate_entries.append(gate_entry)
                gate_entry.grid(row=27, column=i + 1, padx=(10, 0), pady=2, sticky="w")
                gate_entry.bind("<KeyRelease>", update_gate_entries)
            else:
                gate_entries[i].grid(
                    row=27, column=i + 1, padx=(10, 0), pady=2, sticky="w"
                )
                gate_entries[i].bind(
                    "<KeyRelease>", update_gate_entries
                )  # Ensure binding
            if axis_entries[i]:
                axis_entries[i].grid_remove()
            if beta_entries[i]:
                beta_entries[i].grid_remove()
            if phi_entries[i]:
                phi_entries[i].grid_remove()

    elif method == "Phi and Beta":
        phi_label.config(text="Phi:")  # Update to Phi
        beta_label.config(text="Beta (deg.):")
        phi_label.grid()
        beta_label.grid()
        axis_label.grid_remove()
        gate_label.grid_remove()
        for i in range(qubit_count):
            phi_entries[i].grid(row=27, column=i + 1, padx=(10, 0), pady=2, sticky="w")
            beta_entries[i].grid(row=28, column=i + 1, padx=(10, 0), pady=2, sticky="w")
            if axis_entries[i]:
                axis_entries[i].grid_remove()
            if gate_entries[i]:
                gate_entries[i].grid_remove()


def add_qubit():
    global qubit_count
    qubit_count += 1

    # Place the qubit label and selection box
    ttk.Label(root, text=f"Qubit {qubit_count}:").grid(row=0, column=qubit_count)
    qubits_var = tk.StringVar(value=f"q{qubit_count}")
    qubits_vars.append(qubits_var)
    ttk.Combobox(root, textvariable=qubits_var, values=[f"q{qubit_count}"]).grid(
        row=1, column=qubit_count
    )

    # Add parameter entries for the qubit
    Omega_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    Omega_entries.append(Omega_entry)
    Omega_entry.grid(row=2, column=qubit_count)

    sigma_Omega_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    sigma_Omega_entries.append(sigma_Omega_entry)
    sigma_Omega_entry.grid(row=3, column=qubit_count)

    omega_1_max_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    omega_1_max_entries.append(omega_1_max_entry)
    omega_1_max_entry.grid(row=4, column=qubit_count)

    # Add pulse duration entry
    pulse_duration_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    pulse_duration_entries.append(pulse_duration_entry)
    pulse_duration_entry.grid(row=6, column=qubit_count)
    pulse_duration_entry.bind("<KeyRelease>", update_pulse_duration_entries)

    point_in_pulse_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)
    point_in_pulse_entries.append(point_in_pulse_entry)
    point_in_pulse_entry.grid(row=7, column=qubit_count)
    point_in_pulse_entry.bind("<KeyRelease>", update_point_in_pulse_entries)

    wf_type_var = tk.StringVar(value="cheb")
    wf_type_vars.append(wf_type_var)
    ttk.Combobox(
        root,
        textvariable=wf_type_var,
        values=["poly", "fou", "cheb", "hermite", "leg", "chirp", "gegen"],
    ).grid(row=8, column=qubit_count)

    wf_mode_var = tk.StringVar(value="cart")
    wf_mode_vars.append(wf_mode_var)
    ttk.Combobox(
        root, textvariable=wf_mode_var, values=["cart", "polar", "polar_phase"]
    ).grid(row=9, column=qubit_count)

    amplitude_envelope_var = tk.StringVar(value="gn")
    amplitude_envelope_vars.append(amplitude_envelope_var)
    ttk.Combobox(
        root, textvariable=amplitude_envelope_var, values=["hs", "gn", "quad"]
    ).grid(row=10, column=qubit_count)

    amplitude_order_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)
    amplitude_order_entries.append(amplitude_order_entry)
    amplitude_order_entry.grid(row=11, column=qubit_count)

    coverage_var = tk.StringVar(value="broadband")
    coverage_vars.append(coverage_var)
    coverage_combobox = ttk.Combobox(
        root,
        textvariable=coverage_var,
        values=["single", "broadband", "selective", "band_selective"],
    )
    coverage_combobox.grid(row=12, column=qubit_count)
    coverage_combobox.bind("<<ComboboxSelected>>", lambda e: update_coverage_fields())

    sw_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    sw_entries.append(sw_entry)
    sw_entry.grid(row=13, column=qubit_count)

    pulse_offset_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    pulse_offset_entries.append(pulse_offset_entry)
    pulse_offset_entry.grid(row=14, column=qubit_count)

    pulse_bandwidth_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)
    pulse_bandwidth_entries.append(pulse_bandwidth_entry)
    pulse_bandwidth_entry.grid(row=15, column=qubit_count)

    sigma_omega_1_max_entry = ttk.Entry(
        root, validate="key", validatecommand=vcmd_float
    )
    sigma_omega_1_max_entries.append(sigma_omega_1_max_entry)
    sigma_omega_1_max_entry.grid(row=5, column=qubit_count)

    order_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)
    order_entries.append(order_entry)
    order_entry.grid(row=19, column=qubit_count)

    n_para_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)
    n_para_entries.append(n_para_entry)
    n_para_entry.grid(row=20, column=qubit_count)

    # Add initial states for the new qubit
    init_ax_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_string)
    init_ax_entries.append(init_ax_entry)
    init_ax_entry.grid(row=21, column=qubit_count)

    # Add target states entries but don't show them yet
    axis_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_string)
    axis_entries.append(axis_entry)

    beta_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_string)
    beta_entries.append(beta_entry)

    phi_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_string)
    phi_entries.append(phi_entry)

    gate_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_string)
    gate_entries.append(gate_entry)

    update_target_fields()  # Ensure the correct fields are shown initially
    add_coupling_entries()  # Add coupling entries
    update_coverage_fields()  # Show/hide pulse_bandwidth and order based on coverage


def set_default_values():
    default_values = {
        "qubits": [f"q{i + 1}" for i in range(len(qubits_vars))],
        "Delta": [10e6 for _ in range(len(qubits_vars))],
        "sigma_Delta": [0.0 for _ in range(len(qubits_vars))],
        "H0_snapshots": [100] * len(qubits_vars),
        "Omega_R_max": [40e6] * len(qubits_vars),
        "pulse_duration": 200e-9,
        "point_in_pulse": [100] * len(qubits_vars),
        "wf_type": ["cheb"] * len(qubits_vars),
        "wf_mode": ["cart"] * len(qubits_vars),
        "amplitude_envelope": ["gn"] * len(qubits_vars),
        "amplitude_order": [1] * len(qubits_vars),
        "coverage": ["single"] * len(qubits_vars),
        "sw": [5e6] * len(qubits_vars),
        "pulse_offset": [0] * len(qubits_vars),
        "pulse_bandwidth": [5e5] * len(qubits_vars),
        "sigma_Omega_R_max": [0] * len(qubits_vars),
        "profile_order": [2] * len(qubits_vars),
        "n_para": [16] * len(qubits_vars),
        "init_ax": ["Z"] * len(qubits_vars),
        "axis_value": "Z",  # Default Axis value for Axis method
        "phi_value": ["x"]
        * len(qubits_vars),  # Default Phi value for Phi and Beta method
        "beta_value": [180.0]
        * len(qubits_vars),  # Default Beta value for Phi and Beta method
        "gate_value": ["X"]
        if len(qubits_vars) == 1
        else ["CNOT"] * 2
        if len(qubits_vars) == 2
        else ["Toff"] * 3
        if len(qubits_vars) == 3
        else ["X"]
        * len(qubits_vars),  # Default Gate value based on the number of qubits
        "algorithm": "newton-cg",
        "h0_snapshots": 100,
        "Omega_R_snapshots": 1,
        "max_iter": 1000,
        "targ_fid": 0.999,
        "coupling_type": "XY",  # Default value for Coupling Type
        "sigma_J": 0,  # Default value for σ J
    }

    for i in range(len(qubits_vars)):
        qubits_vars[i].set(default_values["qubits"][i])
        Omega_entries[i].delete(0, tk.END)
        Omega_entries[i].insert(0, default_values["Delta"][i])
        sigma_Omega_entries[i].delete(0, tk.END)
        sigma_Omega_entries[i].insert(0, default_values["sigma_Delta"][i])
        omega_1_max_entries[i].delete(0, tk.END)
        omega_1_max_entries[i].insert(0, default_values["Omega_R_max"][i])
        pulse_duration_entries[i].delete(0, tk.END)
        pulse_duration_entries[i].insert(0, default_values["pulse_duration"])
        point_in_pulse_entries[i].delete(0, tk.END)
        point_in_pulse_entries[i].insert(0, default_values["point_in_pulse"][i])
        wf_type_vars[i].set(default_values["wf_type"][i])
        wf_mode_vars[i].set(default_values["wf_mode"][i])
        amplitude_envelope_vars[i].set(default_values["amplitude_envelope"][i])
        amplitude_order_entries[i].delete(0, tk.END)
        amplitude_order_entries[i].insert(0, default_values["amplitude_order"][i])
        coverage_vars[i].set(default_values["coverage"][i])
        sw_entries[i].delete(0, tk.END)
        sw_entries[i].insert(0, default_values["sw"][i])
        pulse_offset_entries[i].delete(0, tk.END)
        pulse_offset_entries[i].insert(0, default_values["pulse_offset"][i])
        pulse_bandwidth_entries[i].delete(0, tk.END)
        pulse_bandwidth_entries[i].insert(0, default_values["pulse_bandwidth"][i])
        sigma_omega_1_max_entries[i].delete(0, tk.END)
        sigma_omega_1_max_entries[i].insert(0, default_values["sigma_Omega_R_max"][i])
        order_entries[i].delete(0, tk.END)
        order_entries[i].insert(0, default_values["profile_order"][i])
        n_para_entries[i].delete(0, tk.END)
        n_para_entries[i].insert(0, default_values["n_para"][i])
        init_ax_entries[i].delete(0, tk.END)
        init_ax_entries[i].insert(0, default_values["init_ax"][i])

        # Set Axis or Phi values based on the selected method
        if target_method_var.get() == "Axis":
            if axis_entries[i]:
                axis_entries[i].delete(0, tk.END)
                axis_entries[i].insert(0, default_values["axis_value"])

        elif target_method_var.get() == "Phi and Beta":
            if phi_entries[i]:
                phi_entries[i].delete(0, tk.END)
                phi_entries[i].insert(0, default_values["phi_value"][i])
            if beta_entries[i]:
                beta_entries[i].delete(0, tk.END)
                beta_entries[i].insert(0, default_values["beta_value"][i])

        elif target_method_var.get() == "Gate":
            if gate_entries[i]:
                gate_entries[i].delete(0, tk.END)
                gate_entries[i].insert(0, default_values["gate_value"][i])

    if qubit_count > 1:  # Ensure that there are multiple qubits
        # Reset the Coupling Type to its default value
        coupling_type_var.set(default_values["coupling_type"])

        # Set default values for J entries
        default_J_values = [
            [
                0.0 if i == j else 10e6 * ((i + 1) * (j + 1) / 1.2)
                for j in range(qubit_count)
            ]
            for i in range(qubit_count)
        ]
        for i, j, entry in coupling_entries:
            entry.delete(0, tk.END)
            entry.insert(0, default_J_values[i - 1][j - 1])

        sigma_J_entry.delete(0, tk.END)
        sigma_J_entry.insert(0, str(default_values["sigma_J"]))

    algorithm_var.set(default_values["algorithm"])
    H0_snapshots_entry.delete(0, tk.END)
    H0_snapshots_entry.insert(0, default_values["h0_snapshots"])
    omega_1_snapshots_entry.delete(0, tk.END)
    omega_1_snapshots_entry.insert(0, default_values["Omega_R_snapshots"])
    max_iter_entry.delete(0, tk.END)
    max_iter_entry.insert(0, default_values["max_iter"])
    targ_fid_entry.delete(0, tk.END)
    targ_fid_entry.insert(0, default_values["targ_fid"])

    # Default compute resource
    compute_resource_var.set("cpu")


# Initialize main window
root = tk.Tk()
root.title("Quantum Control Parameters")

vcmd_int = (root.register(validate_integer), "%P")
vcmd_float = (root.register(validate_float), "%P")
vcmd_string = (root.register(validate_string), "%P")

# Initialize variables
qubits_vars = []
Omega_entries = []
sigma_Omega_entries = []
omega_1_max_entries = []
pulse_duration_entries = []
point_in_pulse_entries = []
wf_type_vars = []
wf_mode_vars = []
amplitude_envelope_vars = []
amplitude_order_entries = []
coverage_vars = []
sw_entries = []
pulse_offset_entries = []
pulse_bandwidth_entries = []
sigma_omega_1_max_entries = []
order_entries = []
n_para_entries = []
init_ax_entries = []  # List to store initial state entries
axis_entries = []  # List to store axis entries for "Axis" or "Phi" method
beta_entries = []  # List to store beta entries for "Phi and Beta" method
phi_entries = []  # List to store phi entries for "Phi and Beta" method
gate_entries = []  # List to store gate entries for "Gate" method
qubit_count = 0  # Initialize qubit count

# Global variables to store the last run results
last_solution = None
last_parameters = None
last_timestamp = None

# Initialize target method variable
target_method_var = tk.StringVar(value="Axis")  # Default to "Axis"

# Initialize algorithm variable
algorithm_var = tk.StringVar(value="newton-cg")  # Default algorithm value

# Initialize space variable
space_var = tk.StringVar(value="hilbert")  # Default space value

# Compute resource selection (cpu/gpu)
compute_resource_var = tk.StringVar(value="cpu")

# Add labels for parameter names
ttk.Label(root, text="Parameter").grid(row=0, column=0)
ttk.Label(root, text="Qubits:").grid(row=1, column=0)
ttk.Label(root, text="Δ (Hz):").grid(row=2, column=0)
ttk.Label(root, text="σΔ (Hz):").grid(row=3, column=0)
ttk.Label(root, text="Ω_R Max (Hz):").grid(row=4, column=0)
ttk.Label(root, text="σΩ_R Max (Hz):").grid(row=5, column=0)
ttk.Label(root, text="Pulse Duration (sec):").grid(row=6, column=0)
ttk.Label(root, text="Point in Pulse:").grid(row=7, column=0)
ttk.Label(root, text="Waveform Type:").grid(row=8, column=0)
ttk.Label(root, text="Waveform Mode:").grid(row=9, column=0)
ttk.Label(root, text="Amplitude Envelope:").grid(row=10, column=0)
ttk.Label(root, text="Amplitude Order:").grid(row=11, column=0)
ttk.Label(root, text="Coverage:").grid(row=12, column=0)
ttk.Label(root, text="Sweep Rate (Hz):").grid(row=13, column=0)
ttk.Label(root, text="Pulse Offset (Hz):").grid(row=14, column=0)
pulse_bandwidth_label = ttk.Label(root, text="Pulse Bandwidth (Hz):")
pulse_bandwidth_label.grid(row=15, column=0)
order_label = ttk.Label(root, text="Profile Order:")
order_label.grid(row=19, column=0)
ttk.Label(root, text="number of parameters:").grid(row=20, column=0)
ttk.Label(root, text="Initial Axis States:").grid(row=21, column=0)
ttk.Label(root, text="Target States Method:").grid(row=22, column=0)

# Dynamic labels for Axis/Phi/Gate
axis_label = ttk.Label(root, text="Axis:")  # Start with Axis as the default
axis_label.grid(row=27, column=0)
beta_label = ttk.Label(root, text="Beta:")
beta_label.grid(row=28, column=0)
phi_label = ttk.Label(root, text="Phi:")
phi_label.grid(row=27, column=0)
gate_label = ttk.Label(root, text="Gate:")
gate_label.grid(row=27, column=0)

# Add Target States Method combobox
target_method_combobox = ttk.Combobox(
    root,
    textvariable=target_method_var,
    values=["Axis", "Gate", "Phi and Beta"],  # Include the new "Phi and Beta" option
)
target_method_combobox.grid(row=22, column=1, columnspan=qubit_count + 1)
target_method_combobox.bind("<<ComboboxSelected>>", lambda e: update_target_fields())

# Initialize coupling frame
coupling_frame = ttk.Frame(root)

# Add initial qubit
add_qubit()

# Define a right-side column for Optimization and Coupling sections
OPT_COL = qubit_count + 55

# Place coupling frame on the right column (will be populated only if >1 qubit)
coupling_frame.grid(row=10, column=OPT_COL, rowspan=32, columnspan=2, sticky="ne")
# Align coupling frame internal columns with optimization frame
coupling_frame.grid_columnconfigure(0, weight=0)
coupling_frame.grid_columnconfigure(1, weight=1)

# Initialize other required variables before calling set_default_values

H0_snapshots_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)
omega_1_snapshots_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)

# Initialize max iterations entry
max_iter_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_int)

# Initialize target fidelity entry
targ_fid_entry = ttk.Entry(root, validate="key", validatecommand=vcmd_float)

# Set default values initially
set_default_values()

# Add "Add Qubit" button
add_qubit_button = ttk.Button(root, text="Add Qubit", command=add_qubit)
add_qubit_button.grid(row=30, column=0, columnspan=4)

# Add optimization parameters section on the top-right
ttk.Label(root, text="Optimization Parameters").grid(
    row=0, column=OPT_COL, columnspan=2, sticky="e"
)

# Space selection
ttk.Label(root, text="Space:").grid(row=1, column=OPT_COL, sticky="e")
ttk.Combobox(root, textvariable=space_var, values=["hilbert", "liouville"]).grid(
    row=1, column=OPT_COL + 1, sticky="e"
)

# H0 snapshots
ttk.Label(root, text="H₀ Snapshots:").grid(row=2, column=OPT_COL, sticky="e")
H0_snapshots_entry.grid(row=2, column=OPT_COL + 1, sticky="e")

# Omega_1 snapshots
ttk.Label(root, text="Ω_R Snapshots:").grid(row=3, column=OPT_COL, sticky="e")
omega_1_snapshots_entry.grid(row=3, column=OPT_COL + 1, sticky="e")

# Algorithm selection
ttk.Label(root, text="Algorithm:").grid(row=4, column=OPT_COL, sticky="e")

# Get supported algorithms dynamically
qiskit_optimizers = get_supported_qiskit_optimizers()
supported_algorithms = [
    "bfgs",
    "l-bfgs",
    "cg",
    "newton-cg",
    "newton-exact",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
] + qiskit_optimizers

ttk.Combobox(
    root,
    textvariable=algorithm_var,
    values=supported_algorithms,
).grid(row=4, column=OPT_COL + 1, sticky="e")

# Max iterations
ttk.Label(root, text="Max Iterations:").grid(row=5, column=OPT_COL, sticky="e")
max_iter_entry.grid(row=5, column=OPT_COL + 1, sticky="e")

# Target fidelity
ttk.Label(root, text="Target Fidelity:").grid(row=6, column=OPT_COL, sticky="e")
targ_fid_entry.grid(row=6, column=OPT_COL + 1, sticky="e")

# Compute Resource selection
ttk.Label(root, text="Compute Resource:").grid(row=7, column=OPT_COL, sticky="e")
ttk.Combobox(root, textvariable=compute_resource_var, values=["cpu", "gpu"]).grid(
    row=7, column=OPT_COL + 1, sticky="e"
)

# Add submit button (left column as before)
submit_button = ttk.Button(root, text="Save", command=save_to_json)
submit_button.grid(row=40, column=0, columnspan=4)

# Add "Default Values" button (left column as before)
default_values_button = ttk.Button(
    root, text="Default Values", command=set_default_values
)
default_values_button.grid(row=41, column=0, columnspan=4)

# Add "Run" button (left column as before)
run_button = ttk.Button(root, text="Run", command=run_optimisation_and_plot)
run_button.grid(row=42, column=0, columnspan=4)

# Add "Save Results" button (left column as before)
save_results_button = ttk.Button(root, text="Save Results", command=save_results)
save_results_button.grid(row=43, column=0, columnspan=4)

# Run the application only if this module is executed directly
if __name__ == "__main__":
    root.mainloop()
