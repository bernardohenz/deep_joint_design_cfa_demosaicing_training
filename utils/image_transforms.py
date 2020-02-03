import numpy as np

def random_flip(x):
    #horizontal flip (x-axis)
    if np.random.random() < 0.5:
        x = np.flip(x, axis=0)
    #vertical flip (y-axis)
    if np.random.random() < 0.5:
        x = np.flip(x, axis=1)
    return x

def random_rot90(x):
    x = np.rot90(x,np.random.randint(4),axes=(0,1))
    return x

def random_shift(x,ranges=(30,30)):
    #x-shift
    rangex,rangey = ranges
    amount_shift = np.random.randint(-rangex, rangex)
    if np.random.random() < 0.5:
        x = np.roll(x,amount_shift,axis=0)
    #y-shift
    amount_shift = np.random.randint(-rangey, rangey)
    if np.random.random() < 0.5:
        x = np.roll(x,amount_shift,axis=0)
    return x

def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size):
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]
