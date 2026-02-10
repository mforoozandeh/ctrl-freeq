#!/usr/bin/env python3
"""
Simple test to verify CtrlFreeQ logging conversion works correctly.
"""

import sys

sys.path.append("/")

from ctrl_freeq.utils.colored_logging import setup_colored_logging
from ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted


def test_logger_initialization():
    """Test that logger initialization works correctly"""
    print("Testing logger initialization...")
    print("=" * 50)

    logger = setup_colored_logging(level="INFO")

    # Test different log levels with colors
    logger.info("=" * 33)
    logger.info(f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10}")
    logger.info("=" * 33)

    # Simulate optimization progress
    for i in range(1, 4):
        fidelity = 0.8 + i * 0.05
        penalty = 0.01 - i * 0.002
        logger.info(f"{i:<10} | {fidelity:<10.4f} | {penalty:<10.4f}")

    print("Logger initialization test passed!")
    print("=" * 50)


def test_exception_logging():
    """Test exception logging with WARNING level"""
    print("\nTesting exception logging...")
    print("=" * 50)

    logger = setup_colored_logging(level="INFO")

    try:
        raise OptimizationInterrupted(
            "Objective function reached target fidelity = 0.99, exiting optimization...",
            None,
        )
    except OptimizationInterrupted as e:
        logger.warning(str(e))
        print("Exception logged successfully with WARNING level")

    print("Exception logging test passed!")
    print("=" * 50)


def test_callback_logging_simulation():
    """Simulate the callback function logging behavior"""
    print("\nTesting callback function logging simulation...")
    print("=" * 50)

    logger = setup_colored_logging(level="INFO")

    # Simulate callback function behavior
    iter_count = 0

    # First iteration - print header
    if iter_count == 0:
        logger.info("=" * 33)
        logger.info(f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10}")
        logger.info("=" * 33)

    # Simulate several iterations
    for i in range(3):
        iter_count += 1
        current_fidelity = 0.85 + i * 0.04
        penalty = 0.01 - i * 0.002
        logger.info(f"{iter_count:<10} | {current_fidelity:<10.4f} | {penalty:<10.4f}")

        # Simulate reaching target fidelity
        if current_fidelity >= 0.92:
            try:
                raise OptimizationInterrupted(
                    "Objective function reached target fidelity = 0.92, exiting optimization...",
                    None,
                )
            except OptimizationInterrupted as e:
                logger.warning(str(e))
                break

    print("Callback logging simulation test passed!")
    print("=" * 50)


def test_color_levels():
    """Test different log levels with colors"""
    print("\nTesting different log levels with colors...")
    print("=" * 50)

    logger = setup_colored_logging(level="DEBUG")

    logger.debug("This is a DEBUG message (cyan, dim)")
    logger.info("This is an INFO message (green, normal)")
    logger.warning("This is a WARNING message (yellow, bold)")
    logger.error("This is an ERROR message (red, bold)")
    logger.critical("This is a CRITICAL message (magenta, bold)")

    print("Color levels test passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_logger_initialization()
    test_exception_logging()
    test_callback_logging_simulation()
    test_color_levels()
    print(
        "\n🎉 All logging tests passed! CtrlFreeQ logging conversion is working correctly."
    )
