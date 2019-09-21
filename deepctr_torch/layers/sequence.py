import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
class KMaxPooling(nn.Module):
    """K Max pooling that selects the k biggest value along the specific axis.

      Input shape
        -  nD tensor with shape: ``(batch_size, ..., input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., output_dim)``.

      Arguments
        - **k**: positive integer, number of top elements to look for along the ``axis`` dimension.

        - **axis**: positive integer, the dimension to look for elements.

     """

    def __init__(self, k=1, axis=-1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.k = k
        self.axis = axis
        


    def forward(self, inputs):
        input_shape = inputs.shape
        if self.axis < 1 or self.axis > len(input_shape):
            raise ValueError("axis must be 1~%d,now is %d" %
                             (len(input_shape), len(input_shape)))

        if self.k < 1 or self.k > input_shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (input_shape[self.axis], self.k))
        dims = len(input_shape)
        perm = list(range(dims))
        perm[-1], perm[self.axis] = perm[self.axis], perm[-1]
        shifted_input = inputs.permute(perm)
        top_k = torch.topk(shifted_input, k=self.k, sorted=True)[0]
        output = top_k.permute(perm)
        return output
