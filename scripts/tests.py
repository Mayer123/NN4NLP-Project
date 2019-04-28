import numpy as np
import torch.nn.functional as F 

def log_softmax(x, axis):
    ret = x - np.max(x, axis=axis, keepdims=True)
    lsm = np.log(np.sum(np.exp(ret), axis=axis, keepdims=True))
    return ret - lsm


def array_to_str(arr, vocab):
    return " ".join(vocab[a] for a in arr)


def test_prediction(out, targ):
    # out = torch.nn.log_softmax(out, 1)
    # nlls = out[np.arange(out.shape[0]), targ]
    # nll = -np.mean(nlls)
    print(out.shape, targ.shape)
    nll = F.nll_loss(F.log_softmax(out, 1), targ.transpose(0,1))
    return nll

def test_generation(inp, pred, vocab):
    outputs = u""
    for i in range(inp.shape[0]):
        w1 = array_to_str(inp[i], vocab)
        w2 = array_to_str(pred[i], vocab)
        outputs += u"Input | Output #{}: {} | {}\n".format(i, w1, w2)
    return outputs