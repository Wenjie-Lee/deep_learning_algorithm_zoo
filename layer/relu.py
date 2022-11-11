import numpy as np


''' relu naive version '''
def relu1(x):
    # x shape = [b, ch, h, w]
    b, ch, h, w = x.shape

    # 1
    out = (np.abs(x) + x) / 2
    return out

def relu2(x):
    # x shape = [b, ch, h, w]
    b, ch, h, w = x.shape

    # 2
    out = np.maximum(x, 0)
    # out = torch.clamp(x, min=0)   # pytorch op
    return out

''' Test '''
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    x = np.random.rand(1, 3, 4, 4) * 2 - 1
    print('img:', x, x.shape)

    out1 = relu1(x)
    print('out1', out1, out1.shape)
    out2 = relu2(x)
    print('out2', out2, out2.shape)

    print('out1 = out2?:', (out1==out2).all())