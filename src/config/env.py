import os

os.environ["PATH"] = os.pathsep.join(
    ["/usr/bin", os.environ.get("PATH", "")]
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt


def initialize_environment():
    """
    Sets up system-level environment variables.
    Call this at the very top of your main entrypoint cli.
    """
    # Disable external LaTeX usage
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",  # Computer Modern font (looks like LaTeX)
    })
