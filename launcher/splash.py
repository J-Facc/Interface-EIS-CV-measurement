"""Fenêtre de chargement affichée pendant le démarrage."""
import tkinter as tk

_root = None
_label = None


def update_label(msg: str):
    global _label, _root
    if _label:
        _label.config(text=msg)
    if _root:
        _root.update_idletasks()


def show_splash():
    global _root, _label
    _root = tk.Tk()
    _root.title("EIS Analyzer")
    _root.geometry("420x130")
    _root.resizable(False, False)
    _root.configure(bg="#1a1a2e")
    _root.overrideredirect(True)

    # Centrer sur l'écran
    _root.update_idletasks()
    x = (_root.winfo_screenwidth() - 420) // 2
    y = (_root.winfo_screenheight() - 130) // 2
    _root.geometry(f"+{x}+{y}")

    tk.Label(_root, text="⚡ EIS Analyzer",
             font=("Helvetica", 16, "bold"),
             bg="#1a1a2e", fg="#4fc3f7").pack(pady=(22, 6))

    _label = tk.Label(_root, text="Démarrage...",
                      font=("Helvetica", 10),
                      bg="#1a1a2e", fg="#E0E0E0")
    _label.pack()
    _root.mainloop()
