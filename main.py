import tkinter as tk
from pencil_sketch_app.ui.app_window import PencilSketchApp


def main() -> None:
    root = tk.Tk()
    PencilSketchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
