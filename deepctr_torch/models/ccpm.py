# -*- coding:utf-8 -*-
"""

Author:
    Zeng Kai,kk163mail@126.com

Reference:
    [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
    (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from ..layers.utils import Conv2dSame
from ..layers.core import DNN
from ..layers.sequence import KMaxPooling
from ..layers.utils import concat_fun
from .basemodel import BaseModel

class CCPM(BaseModel):
    """Instantiates the Convolutional Click Prediction Model architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, embedding_size=8, conv_kernel_width=(6, 5), conv_filters=(4, 4),
         dnn_hidden_units=(256,), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
         init_std=0.0001, seed=1024, task='binary',device='cpu',dnn_use_bn=False,dnn_activation=F.relu):

        super(CCPM, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                      dnn_hidden_units=dnn_hidden_units,
                                      l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                      seed=seed,
                                      dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                      task=task, device=device)
         
        if len(conv_kernel_width) != len(conv_filters):
            raise ValueError(
                "conv_kernel_width must have same element with conv_filters")

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std,device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

    def forward(self,X):

        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_logit = self.linear_model(X)
        conv_input = concat_fun(sparse_embedding_list, axis=1)
        pooling_result = filter(lambda x: torch.unsqueeze(x, 3),conv_input)
        n = len(sparse_embedding_list)
        l = len(conv_filters)
        
        for i in range(1, l + 1):
            filters = conv_filters[i - 1]
            width = conv_kernel_width[i - 1]
            k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3
            conv_result = Conv2dSame(in_channels=pooling_result.shape[-1],out_channels=filters,kernel_size=(width, 1),stride=(1,1))(pooling_result)
            conv_result = torch.nn.functional.tanh()(conv_result)
            pooling_result = KMaxPooling(
                k=min(k, int(conv_result.shape[1])), axis=1)(conv_result)

        flatten_result = pooling_result.view(pooling_result.size(0), -1)
        dnn_output = self.dnn(flatten_result)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = linear_logit + dnn_logit
        y_pred = self.out(logit)
        return y_pred





