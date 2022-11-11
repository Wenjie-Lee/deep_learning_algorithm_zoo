import numpy as np


''' batch normalization naive version '''
def batchnorm2d(x, gamma, beta, running_mean, running_var, mode='train'):
    # x shape = [b, ch, h, w]
    b, ch, h, w = x.shape
    eps = 1e-8
    momentum = 0.9  # 更新动量

    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    out, cache = None, None
    if mode == 'train':
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta
        cache = (gamma, x, mean, var, eps, x_hat)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
    elif mode == 'test':
        scale = gamma / (np.sqrt(running_var  + eps))
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache