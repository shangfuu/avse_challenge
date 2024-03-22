import keras.ops as ops


def pad(x, stride):
    h, w = x.shape[1:3]
    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = ([0, 0], [lh, uh], [lw, uw], [0, 0])
    out = ops.pad(x, pads, "constant", 0)
    return out, pads


def unpad(x, pad):
    [_, _], [lw, uw], [lh, uh], [_, _] = pad
    if lh + uh > 0:
        x = x[:, :, lh:-uh, :]
    if lw + uw > 0:
        x = x[:, lw:-uw, :, :]
    return x
