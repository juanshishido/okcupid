import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def wcloud(wf, color, save_as=None):
    """Create a word cloud based on word frequencies,
    `wf`, using a color function from `wc_colors.py`

    Parameters
    ----------
    wf : list
        (token, value) tuples
    color : function
        from `wc_colors.py`
    save_as : str
        filename

    Returns
    -------
    None
    """
    wc = WordCloud(background_color=None, mode='RGBA',
                   width=2400, height=1600, relative_scaling=0.5,
                   font_path='/Library/Fonts/Futura.ttc')
    wc.generate_from_frequencies(wf)
    plt.figure()
    plt.imshow(wc.recolor(color_func=color, random_state=42))
    plt.axis("off")
    if save_as:
        plt.savefig(save_as, dpi=300, transparent=True)

def lollipop(df, demographic, colors):
    """Create the lollipop plots for the percentage of users in each NMF group

    Parameters
    ----------
    df : pd.DataFrame
        Should be created using `group_pct()` in `utils/splits.py`
    demographic : str
        Valid column name
    colors : list
        Valid Matplotlib colors codes or names (e.g., hex)

    Returns
    -------
    None
    """
    df = df.copy()
    # styling
    if not colors:
        colors = ['#348ABD', '#A60628', '#7A68A6', '#467821',
                  '#D55E00', '#CC79A7', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#A500FF', '#FFA500']
    sns.set_style("dark")
    fs = 28
    weight = 'bold'
    text_color = 'lightgray'
    # plot
    fig, ax = plt.subplots(figsize=(12, 8))
    # lines
    lineval = df.groupby('group')['pct'].max()
    for i, g in enumerate(lineval):
        plt.plot([i, i], [0, g], linewidth=10,
                 color='lightgray', zorder=1)
    # markers
    for i, d in enumerate(df[demographic].unique()):
        tdf = df[df[demographic]==d]
        plt.scatter(range(len(tdf)), tdf.pct, s=400,
                    color=colors[i], edgecolor='lightgray', lw=4,
                    zorder=2, label=d.capitalize())
    # plot options
    plt.xlim(-0.5, len(tdf)-0.5)
    plt.ylim(0)
    plt.gca().get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda y, p: format(y, '.0%')))
    plt.xlabel('Group')
    plt.ylabel('Normalized Percentage of Users')
    lg = ax.legend(title=demographic.title(), fontsize=fs, loc='upper right',
                   bbox_to_anchor=(1.15, 1))
    for text in lg.get_texts():
        plt.setp(text, color=text_color, weight=weight)
    lg.get_title().set_fontweight(weight)
    lg.get_title().set_color(text_color)
    lg.get_title().set_fontsize(fs)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight(weight)
        label.set_fontsize(fs)
        label.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.xaxis.label.set_fontweight(weight)
    ax.xaxis.label.set_fontsize(fs)
    ax.yaxis.label.set_color(text_color)
    ax.yaxis.label.set_fontweight(weight)
    ax.yaxis.label.set_fontsize(fs)
