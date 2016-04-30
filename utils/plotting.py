import matplotlib.pyplot as plt
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
