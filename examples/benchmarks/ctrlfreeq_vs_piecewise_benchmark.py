"""
Benchmark script to compare CtrlFreeQ (orthogonal basis) vs Piecewise optimization methods.

This script provides a fair comparison between the two approaches by:
1. Testing the same quantum control problems with both methods using CtrlFreeQ API
2. Comparing fidelity vs iteration curves
3. Analyzing parameter efficiency (number of parameters to reach target fidelity)
4. Measuring convergence speed and final fidelity achieved

Uses CtrlFreeQ API with JSON configurations as demonstrated in api_demo.ipynb
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union
import json
from datetime import datetime
from pathlib import Path

from src.ctrl_freeq.utils.colored_logging import setup_colored_logging
from src.ctrl_freeq.api import (
    load_config,
    CtrlFreeQAPI,
)
from src.ctrl_freeq.visualisation.plotter import process_and_plot
from src.ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI, pulse_para_piecewise


class BenchmarkResults:
    """Class to store and analyze benchmark results."""

    def __init__(self):
        self.results = {"ctrlfreeq": {}, "piecewise": {}}
        self.metadata = {"timestamp": datetime.now().isoformat(), "test_cases": []}

    def add_result(self, method: str, test_case: str, result: Dict[str, Any]):
        """Add a benchmark result."""
        if test_case not in self.results[method]:
            self.results[method][test_case] = {}
        self.results[method][test_case].update(result)

    def save_results(self, filename: str):
        """Save results to JSON file."""
        with open(filename, "w") as f:
            json.dump(
                {"results": self.results, "metadata": self.metadata},
                f,
                indent=2,
                default=str,
            )

    def plot_fidelity_comparison(self, test_case: str, save_path: str = None):
        """Plot fidelity vs iteration comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"CtrlFreeQ vs Piecewise Optimization Comparison - {test_case}", fontsize=16
        )

        methods = ["cart", "polar", "polar_phase"]

        for i, method in enumerate(methods):
            # Fidelity vs Iteration
            ax1 = axes[0, i]
            if f"{method}_ctrlfreeq" in self.results["ctrlfreeq"][test_case]:
                ctrlfreeq_hist = self.results["ctrlfreeq"][test_case][
                    f"{method}_ctrlfreeq"
                ]["fidelity_history"]
                ax1.plot(ctrlfreeq_hist, label=f"CtrlFreeQ {method}", linewidth=2)

            if f"{method}_piecewise" in self.results["piecewise"][test_case]:
                piece_hist = self.results["piecewise"][test_case][
                    f"{method}_piecewise"
                ]["fidelity_history"]
                ax1.plot(
                    piece_hist, label=f"Piecewise {method}", linewidth=2, linestyle="--"
                )

            ax1.set_title(f"{method.title()} Method")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Fidelity")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Fidelity vs Time
            ax2 = axes[1, i]
            if f"{method}_ctrlfreeq" in self.results["ctrlfreeq"][test_case]:
                ctrlfreeq_result = self.results["ctrlfreeq"][test_case][
                    f"{method}_ctrlfreeq"
                ]
                ctrlfreeq_hist = ctrlfreeq_result["fidelity_history"]
                ctrlfreeq_time = ctrlfreeq_result.get("optimization_time", 1.0)
                # Create time array based on fidelity history length
                time_array = np.linspace(0, ctrlfreeq_time, len(ctrlfreeq_hist))
                ax2.plot(
                    time_array, ctrlfreeq_hist, label=f"CtrlFreeQ {method}", linewidth=2
                )

            if f"{method}_piecewise" in self.results["piecewise"][test_case]:
                piece_result = self.results["piecewise"][test_case][
                    f"{method}_piecewise"
                ]
                piece_hist = piece_result["fidelity_history"]
                piece_time = piece_result.get("optimization_time", 1.0)
                # Create time array based on fidelity history length
                time_array = np.linspace(0, piece_time, len(piece_hist))
                ax2.plot(
                    time_array,
                    piece_hist,
                    label=f"Piecewise {method}",
                    linewidth=2,
                    linestyle="--",
                )

            ax2.set_title(f"{method.title()} Method")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Fidelity")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_optimized_waveforms(
        self, test_case: str, config: Dict[str, Any], save_path: str = None
    ):
        """Plot the final optimized waveforms for both CtrlFreeQ and piecewise methods."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle(f"Final Optimized Waveforms - {test_case}", fontsize=16)

        methods = ["cart", "polar", "polar_phase"]

        for i, method in enumerate(methods):
            # Extract results for this method
            ctrlfreeq_key = f"{method}_ctrlfreeq"
            piecewise_key = f"{method}_piecewise"

            if (
                ctrlfreeq_key in self.results["ctrlfreeq"][test_case]
                and piecewise_key in self.results["piecewise"][test_case]
            ):
                ctrlfreeq_result = self.results["ctrlfreeq"][test_case][ctrlfreeq_key]
                piecewise_result = self.results["piecewise"][test_case][piecewise_key]

                # Generate waveforms from final parameters
                ctrlfreeq_waveforms = ctrlfreeq_result.get("waveforms", {})
                piecewise_waveforms = piecewise_result.get("waveforms", {})

                # Time axis for plotting
                n_pulse = config["parameters"]["point_in_pulse"][0]
                time_steps = np.arange(n_pulse)

                if method == "cart":
                    # For cart method: plot cx and cy as before
                    # Plot cx (real part)
                    ax1 = axes[0, i]
                    if "cx" in ctrlfreeq_waveforms:
                        ax1.plot(
                            time_steps,
                            ctrlfreeq_waveforms["cx"],
                            label="CtrlFreeQ",
                            linewidth=2,
                        )
                    if "cx" in piecewise_waveforms:
                        ax1.plot(
                            time_steps,
                            piecewise_waveforms["cx"],
                            label="Piecewise",
                            linewidth=2,
                            linestyle="--",
                        )
                    ax1.set_title(f"{method.title()} - cx(t)")
                    ax1.set_ylabel("Amplitude")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Plot cy (imaginary part)
                    ax2 = axes[1, i]
                    if "cy" in ctrlfreeq_waveforms:
                        ax2.plot(
                            time_steps,
                            ctrlfreeq_waveforms["cy"],
                            label="CtrlFreeQ",
                            linewidth=2,
                        )
                    if "cy" in piecewise_waveforms:
                        ax2.plot(
                            time_steps,
                            piecewise_waveforms["cy"],
                            label="Piecewise",
                            linewidth=2,
                            linestyle="--",
                        )
                    ax2.set_title(f"{method.title()} - cy(t)")
                    ax2.set_ylabel("Amplitude")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                else:
                    # For polar and polar_phase methods: plot amplitude and phase
                    # Plot amplitude
                    ax1 = axes[0, i]
                    if "amp" in ctrlfreeq_waveforms:
                        ax1.plot(
                            time_steps,
                            ctrlfreeq_waveforms["amp"],
                            label="CtrlFreeQ",
                            linewidth=2,
                        )
                    if "amp" in piecewise_waveforms:
                        ax1.plot(
                            time_steps,
                            piecewise_waveforms["amp"],
                            label="Piecewise",
                            linewidth=2,
                            linestyle="--",
                        )
                    ax1.set_title(f"{method.title()} - Amplitude")
                    ax1.set_ylabel("Amplitude")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Plot phase
                    ax2 = axes[1, i]
                    # Calculate phase from cx and cy
                    if "cx" in ctrlfreeq_waveforms and "cy" in ctrlfreeq_waveforms:
                        ctrlfreeq_phase = np.arctan2(
                            ctrlfreeq_waveforms["cy"], ctrlfreeq_waveforms["cx"]
                        )
                        ax2.plot(
                            time_steps, ctrlfreeq_phase, label="CtrlFreeQ", linewidth=2
                        )
                    if "cx" in piecewise_waveforms and "cy" in piecewise_waveforms:
                        piece_phase = np.arctan2(
                            piecewise_waveforms["cy"], piecewise_waveforms["cx"]
                        )
                        ax2.plot(
                            time_steps,
                            piece_phase,
                            label="Piecewise",
                            linewidth=2,
                            linestyle="--",
                        )
                    ax2.set_title(f"{method.title()} - Phase")
                    ax2.set_ylabel("Phase (rad)")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def run_ctrlfreeq_benchmark(
    config: Union[str, Path, Dict[str, Any]], method: str = "cart"
) -> Dict[str, Any]:
    """Run CtrlFreeQ optimization benchmark using API."""

    # Create CtrlFreeQ API instance
    if isinstance(config, (str, Path)):
        api = load_config(config)
    else:
        api = CtrlFreeQAPI(config.copy())

    # Update method if needed
    if method != "cart":
        api.update_parameter("parameters.wf_mode", [method])

    # Track optimization details
    start_time = time.time()

    # Run optimization using API
    final_params = api.run_optimization()

    end_time = time.time()

    # Generate waveforms for visualization
    waveforms_data, figures = process_and_plot(
        final_params, api.parameters, show_plots=False
    )

    # Get parameter count from API configuration
    if method == "polar_phase":
        n_para = api.config["parameters"]["n_para"][0] + 1
    else:
        n_para = 2 * api.config["parameters"]["n_para"][0]

    total_params = n_para * len(api.config["qubits"])

    # Get fidelity history from API (if available)
    fidelity_history = getattr(api.parameters, "fidelity_history", [])

    # Extract waveforms from the returned data structure
    cx_waveform, cy_waveform, amp_waveform = [], [], []
    if "waveforms" in waveforms_data and waveforms_data["waveforms"]:
        # Use first initial state's waveforms (index 0) and first qubit (index 0)
        first_waveform = waveforms_data["waveforms"][0]
        if "cxs" in first_waveform and "cys" in first_waveform:
            cxs = first_waveform["cxs"]
            cys = first_waveform["cys"]
            if len(cxs) > 0 and len(cys) > 0:
                # For single qubit, take first qubit's waveform
                cx_waveform = cxs[0] if isinstance(cxs, list) and len(cxs) > 0 else cxs
                cy_waveform = cys[0] if isinstance(cys, list) and len(cys) > 0 else cys
                # Convert to numpy arrays if they aren't already
                if hasattr(cx_waveform, "detach"):
                    cx_waveform = cx_waveform.detach().numpy()
                if hasattr(cy_waveform, "detach"):
                    cy_waveform = cy_waveform.detach().numpy()
                # Calculate amplitude
                amp_waveform = np.sqrt(
                    np.array(cx_waveform) ** 2 + np.array(cy_waveform) ** 2
                )

    return {
        "method": f"{method}_ctrlfreeq",
        "converged": True,
        "final_fidelity": float(getattr(api.parameters, "final_fidelity", 0.0)),
        "n_iterations": int(getattr(api.parameters, "iterations", 0)),
        "n_parameters": total_params,
        "optimization_time": end_time - start_time,
        "fidelity_history": fidelity_history,
        "final_params": final_params.detach().numpy().tolist(),
        "waveforms": {
            "cx": cx_waveform.tolist()
            if hasattr(cx_waveform, "tolist")
            else cx_waveform,
            "cy": cy_waveform.tolist()
            if hasattr(cy_waveform, "tolist")
            else cy_waveform,
            "amp": amp_waveform.tolist()
            if hasattr(amp_waveform, "tolist")
            else amp_waveform,
        },
    }


def run_piecewise_benchmark(
    config: Union[str, Path, Dict[str, Any]], method: str = "cart"
) -> Dict[str, Any]:
    """Run piecewise optimization benchmark."""

    # Create piecewise API instance
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            config_dict = json.load(f)
    else:
        config_dict = config.copy()

    piecewise_api = PiecewiseAPI(config_dict, method)

    # Track optimization details
    start_time = time.time()

    # Run optimization
    final_params = piecewise_api.run_optimization()

    end_time = time.time()

    # Get results from piecewise instance
    piecewise_instance = piecewise_api._piecewise_instance

    # Generate waveforms using the actual piecewise waveform pipeline (accurate)
    import torch

    # Split flat parameter vector into per-qubit tensors according to the optimizer instance
    params_per_qubit = []
    idx = 0
    for n in piecewise_instance.n_para:
        params_per_qubit.append(final_params[idx : idx + n].clone().detach())
        idx += n

    # Ensure torch tensors of the correct dtype
    params_per_qubit = [
        (
            p
            if isinstance(p, torch.Tensor)
            else torch.tensor(p, dtype=piecewise_instance.dtype)
        )
        for p in params_per_qubit
    ]

    amps_t, phis_t, cxs_t, cys_t = pulse_para_piecewise(
        n_qubits=len(config_dict["qubits"]),
        parameters=params_per_qubit,
        identity_basis=piecewise_instance.identity_basis,
        wf_fun=piecewise_instance.wf_fun,
        me=piecewise_instance.me,
    )

    # Use the first (and usually only) qubit column for plotting convenience
    cx_waveform = cxs_t[:, 0].detach().cpu().numpy()
    cy_waveform = cys_t[:, 0].detach().cpu().numpy()
    amp_waveform = amps_t[:, 0].detach().cpu().numpy()

    # Calculate parameter count for piecewise method (trust the optimizer instance)
    total_params = int(sum(piecewise_instance.n_para))

    return {
        "method": f"{method}_piecewise",
        "converged": True,
        "final_fidelity": float(piecewise_instance.fid.detach()),
        "n_iterations": int(piecewise_instance.iter),
        "n_parameters": total_params,
        "optimization_time": end_time - start_time,
        "fidelity_history": piecewise_instance.fidelity_history,
        "final_params": final_params.detach().numpy().tolist(),
        "waveforms": {
            "cx": cx_waveform,
            "cy": cy_waveform,
            "amp": amp_waveform,
        },
    }


def run_comprehensive_benchmark(
    test_cases: List[Union[str, Path, Dict[str, Any]]],
    methods: List[str] = ["cart", "polar", "polar_phase"],
    save_results: bool = True,
) -> BenchmarkResults:
    """
    Run comprehensive benchmark comparing CtrlFreeQ vs Piecewise methods using API.

    Args:
        test_cases: List of configuration files or dictionaries
        methods: List of methods to test
        save_results: Whether to save results to file

    Returns:
        BenchmarkResults object with all results
    """
    benchmark = BenchmarkResults()
    logger = setup_colored_logging(level="INFO")

    for i, config in enumerate(test_cases):
        test_name = f"test_case_{i}"

        # Load config to extract information
        if isinstance(config, (str, Path)):
            with open(config, "r") as f:
                config_dict = json.load(f)
        else:
            config_dict = config

        # Extract information from configuration
        n_qubits = len(config_dict["qubits"])
        n_pulse = config_dict["parameters"]["point_in_pulse"][0]
        basis_size = config_dict["parameters"]["n_para"][0]

        benchmark.metadata["test_cases"].append(
            {
                "name": test_name,
                "n_qubits": n_qubits,
                "n_pulse": n_pulse,
                "basis_size": basis_size,
            }
        )

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running benchmark for {test_name}")
        logger.info(
            f"n_qubits: {n_qubits}, n_pulse: {n_pulse}, basis_size: {basis_size}"
        )
        logger.info(f"{'=' * 50}")

        for method in methods:
            try:
                # Run CtrlFreeQ benchmark
                logger.info(f"\nRunning CtrlFreeQ {method} optimization...")
                ctrlfreeq_result = run_ctrlfreeq_benchmark(config, method)
                benchmark.add_result(
                    "ctrlfreeq", test_name, {f"{method}_ctrlfreeq": ctrlfreeq_result}
                )

                # Run Piecewise benchmark
                logger.info(f"\nRunning Piecewise {method} optimization...")
                piecewise_result = run_piecewise_benchmark(config, method)
                benchmark.add_result(
                    "piecewise", test_name, {f"{method}_piecewise": piecewise_result}
                )

                # Log comparison
                logger.info(f"\n{method.upper()} METHOD COMPARISON:")
                logger.info(
                    f"CtrlFreeQ    - Fidelity: {ctrlfreeq_result['final_fidelity']:.6f}, "
                    f"Iterations: {ctrlfreeq_result['n_iterations']}, "
                    f"Parameters: {ctrlfreeq_result['n_parameters']}"
                )
                logger.info(
                    f"Piecewise - Fidelity: {piecewise_result['final_fidelity']:.6f}, "
                    f"Iterations: {piecewise_result['n_iterations']}, "
                    f"Parameters: {piecewise_result['n_parameters']}"
                )

                # Parameter efficiency
                param_ratio = (
                    piecewise_result["n_parameters"] / ctrlfreeq_result["n_parameters"]
                )
                logger.info(
                    f"Parameter ratio (Piecewise/CtrlFreeQ): {param_ratio:.1f}x"
                )

            except Exception as e:
                logger.error(
                    f"Error running {method} benchmark: {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue

        # Plot comparison for this test case
        try:
            # Determine results directory path
            results_dir = Path(__file__).parent.parent.parent / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Check if we have any results for this test case before plotting
            if (
                test_name in benchmark.results["ctrlfreeq"]
                and benchmark.results["ctrlfreeq"][test_name]
            ) or (
                test_name in benchmark.results["piecewise"]
                and benchmark.results["piecewise"][test_name]
            ):
                benchmark.plot_fidelity_comparison(
                    test_name, str(results_dir / f"benchmark_{test_name}.png")
                )
                benchmark.plot_optimized_waveforms(
                    test_name,
                    config_dict,
                    str(results_dir / f"waveforms_{test_name}.png"),
                )
            else:
                logger.warning(f"No results available for plotting {test_name}")
        except Exception as e:
            logger.error(
                f"Error plotting results for {test_name}: {type(e).__name__}: {str(e)}"
            )
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

    if save_results:
        # Save results to the results directory
        results_dir = Path(__file__).parent.parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark.save_results(
            str(results_dir / f"ctrlfreeq_vs_piecewise_benchmark_{timestamp}.json")
        )

    return benchmark


def main():
    """Main benchmark execution using CtrlFreeQ API with JSON configurations."""
    logger = setup_colored_logging(level="INFO")
    logger.info("Starting CtrlFreeQ vs Piecewise Optimization Benchmark")
    logger.info(
        "Using CtrlFreeQ API with JSON configurations as shown in api_demo.ipynb"
    )

    # Get path to JSON configurations
    json_dir = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ctrl_freeq"
        / "data"
        / "json_input"
    )

    # Use existing JSON configurations as test cases
    test_configs = []

    base_config_path = json_dir / "single_qubit_parameters.json"
    if base_config_path.exists():
        # # Test case 1: Modified configuration with different parameters
        # with open(base_config_path, "r") as f:
        #     config1 = json.load(f)
        # config1["parameters"]["point_in_pulse"] = [50]
        # config1["parameters"]["n_para"] = [16]
        # config1["optimization"]["max_iter"] = 3000
        # config1["optimization"]["algorithm"] = "qiskit-cobyla"
        # test_configs.append(config1)
        #
        # # Test case 2: Modified configuration with different parameters
        # with open(base_config_path, "r") as f:
        #     config2 = json.load(f)
        # config2["parameters"]["point_in_pulse"] = [100]
        # config2["parameters"]["n_para"] = [32]
        # config2["optimization"]["max_iter"] = 3000
        # config2["optimization"]["algorithm"] = "qiskit-cobyla"
        # test_configs.append(config2)

        # Test case 3: Modified configuration with different parameters
        with open(base_config_path, "r") as f:
            config2 = json.load(f)
        config2["parameters"]["point_in_pulse"] = [200]
        config2["parameters"]["n_para"] = [32]
        config2["optimization"]["max_iter"] = 1000
        config2["optimization"]["algorithm"] = "qiskit-cobyla"
        test_configs.append(config2)

    if not test_configs:
        logger.error("No JSON configuration files found!")
        return None

    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        test_cases=test_configs,
        methods=["cart", "polar", "polar_phase"],
        save_results=True,
    )

    logger.info(
        "Benchmark for comparison between CtrlFreeQ and piecewise optimization completed successfully!"
    )
    return results


if __name__ == "__main__":
    main()
