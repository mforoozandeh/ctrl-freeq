#!/usr/bin/env python3
"""
Test script to verify the new logging format changes
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ctrl_freeq.utils.colored_logging import setup_colored_logging


def test_logging_formats():
    """Test different log levels to verify formatting"""
    logger = setup_colored_logging(level="DEBUG")

    print("Testing new logging format:")
    print("=" * 50)

    # Test INFO messages (should show only message)
    logger.info("Number of threads set to 12 out of 12 available cores.")
    logger.info("=================================")
    logger.info("Iteration  | Fidelity   | Penalty   ")
    logger.info("=================================")
    logger.info("1          | 0.9269     | 0.0000    ")
    logger.info("2          | 0.9959     | 0.0000    ")
    logger.info("3          | 1.0000     | 0.0000    ")

    print()

    # Test WARNING messages (should show WARNING - message)
    logger.warning("This is a warning message")

    # Test ERROR messages (should show ERROR - message)
    logger.error("This is an error message")

    # Test DEBUG messages (should show full format)
    logger.debug("This is a debug message")

    print()
    print("Test completed!")


if __name__ == "__main__":
    test_logging_formats()
