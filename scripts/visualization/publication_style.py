import matplotlib.pyplot as plt

def set_publication_style():
    """
    Global publication-style settings for all project figures.
    """

    plt.rcParams.update({
        # Font
        "font.family": "DejaVu Sans",
        "font.size": 12,

        # Titles and labels
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.titleweight": "bold",

        # Ticks
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # Legend
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.95,

        # Lines
        "lines.linewidth": 2.5,
        "lines.markersize": 5,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        # Output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axes(ax):
    """
    Apply final cleaning to individual axes.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)


def save_figure(fig, path):
    """
    Save figure in publication quality.
    """
    fig.savefig(path, dpi=600, bbox_inches="tight")