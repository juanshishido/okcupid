import matplotlib as mpl


def gray(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#cccccc')])
    return 'rgb'+str(RGB)

def blue(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#348ABD')])
    return 'rgb'+str(RGB)

def red(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#A60628')])
    return 'rgb'+str(RGB)

def purple(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#7A68A6')])
    return 'rgb'+str(RGB)

def green(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#467821')])
    return 'rgb'+str(RGB)

def orange(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#D55E00')])
    return 'rgb'+str(RGB)

def green_orange(word, font_size, position, orientation, random_state=None, **kwargs):
    green = 'rgb'+str(tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#467821')]))
    orange = 'rgb'+str(tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#D55E00')]))
    if random.randint(0, 1):
        return green
    else:
        return orange
    
def red_blue(word, font_size, position, orientation, random_state=None, **kwargs):
    red = 'rgb'+str(tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#A60628')]))
    blue = 'rgb'+str(tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#348ABD')]))
    if random.randint(0, 1):
        return red
    else:
        return blue
    
def yellow(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#ffd966')])
    return 'rgb'+str(RGB)

def cyan(word, font_size, position, orientation, random_state=None, **kwargs):
    RGB = tuple([round(c * 255) for c in mpl.colors.colorConverter.to_rgb('#64ffda')])
    return 'rgb'+str(RGB)
