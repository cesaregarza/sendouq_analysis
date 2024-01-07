import matplotlib.pyplot as plt
import seaborn as sns


def setup_style() -> None:
    sns.set(style="darkgrid", context="talk")
    sns.color_palette("bright")
    plt.rcParams["axes.facecolor"] = "black"
    plt.rcParams["figure.facecolor"] = "black"

    plt.rcParams["grid.color"] = "gray"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5

    plt.rcParams["axes.edgecolor"] = "white"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["axes.titlecolor"] = "white"

    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"

    plt.rcParams["text.color"] = "white"


class COLORS:
    accent_color = "#ad5ad7"
    accent_darker = "#7a28a3"
    contrast_color = "#228b22"
    purple = "#ad5ad7"
    green = "#228b22"
    blue = "#1e90ff"
    red = "#ff6347"
    orange = "#ffa500"
    yellow = "#ffd700"
    pink = "#ff69b4"
    gray = "#808080"


TIER_COLORS = {
    "leviathan": "#178177",
    "diamond": "#f35c8e",
    "platinum": "#c0c0c0",
    "gold": "#d4af37",
    "silver": "#a8a8a8",
    "bronze": "#cd7f32",
    "iron": "#706e69",
}
