"""Main entry point for running the Malaria Transmission Model Application."""

import tkinter as tk
from gui import MalariaModelApp


def main():
    root = tk.Tk()
    app = MalariaModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
