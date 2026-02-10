#!/usr/bin/env python3
"""
Command-line interface for CtrlFreeQ GUI.
This script provides a command-line entry point to launch the CtrlFreeQ GUI.
"""


def main():
    """Launch the CtrlFreeQ GUI."""

    # Import and run the GUI setup code
    from ctrl_freeq.setup.gui_setup import root

    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()
