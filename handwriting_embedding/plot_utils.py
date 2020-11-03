colour_palette = [
    "#d7191c",
    "#fdae61",
    "#ffffbf",
    "#abdda4",
    "#2b83ba",
]

colour_dict = {
    "text": colour_palette[0],
    "plz": colour_palette[1],
    "alpha_num": colour_palette[2],
    "alphanum": colour_palette[2],
    "date": colour_palette[3],
    "num": colour_palette[4],
    "rest": colour_palette[0]
}

label_dict = {
    "text": "Words",
    "plz": "Zip Codes",
    "alpha_num": "Alpha Numeric",
    "alphanum": "Alpha Numeric",
    "date": "Dates",
    "num": "Numbers",
    "rest": "Others",
}


def configure_plots(plt, size="small"):
    if size == "large":
        SMALL_SIZE = 28
        MEDIUM_SIZE = 32
        BIGGER_SIZE = 36
        LEGEND_SIZE = 26
        lw = 4
    elif size == "medium":
        SMALL_SIZE = 18
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26
        LEGEND_SIZE = 22
        lw = 3
    else:
        SMALL_SIZE = 14
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 24
        LEGEND_SIZE = 16
        lw = 2

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    return lw
