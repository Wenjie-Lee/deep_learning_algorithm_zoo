import numpy as np



''' conv naive version'''
def conv_naive(x, out_ch, kernel=3, padding=0, stride=1):
    # x shape = [b, ch, h, w]
    b, ch, h, w = x.shape

    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(kernel, tuple):
        kh, kw = kernel
    kernel = np.random.rand(kh, kw, ch, out_ch) # params, init only in __init__()

    if padding > 0:
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(padding, tuple):
            ph, pw = padding
        pad_x = np.zeros((b, ch, h+2*ph, w+2*pw))
        pad_x[:, :, ph:-ph, pw:-pw] = x

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(stride, tuple):
        sh, sw = stride
    out_h = (h + 2*ph - kh) // sh + 1
    out_w = (w + 2*pw - kw) // sw + 1
    out = np.zeros((b, out_ch, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            roi_x = pad_x[:, :, i*sh:i*sh+kh, i*sw:i*sw+kw]
            # roi_x shape  = [b, ch, kh, kw] -> [b, out_ch, kh, kw]
            # kernel shape = [kh, kw, ch, out_ch]
            # conv shape   = [b, kw, kh, ch, out_ch] -> [b, 1, 1, out_ch]
            conv = np.tile(np.expand_dims(roi_x, -1), (1,1,1,1,out_ch)) * kernel
            out[:, :, i, j] = np.sum(conv, axis=(1,2,3))
    return out

''' Test '''
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    x = np.random.rand(1, 3, 16, 16)
    print('image:', x, x.shape)

    should_out_x = (16 + 2*1 - 3) // 2 + 1
    should_out_y = (16 + 2*1 - 3) // 2 + 1
    print('should out size:', (1, 3, should_out_x, should_out_y))

    out = conv_naive(x, 16, kernel=3, padding=1, stride=2)
    print('after conv op:', out, out.shape)