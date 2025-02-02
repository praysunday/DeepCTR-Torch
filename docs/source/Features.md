# Features

## Overview

With the great success of deep learning,DNN-based techniques have been widely used in CTR estimation task.

DNN based CTR estimation models consists of the following 4 modules:
`Input,Embedding,Low-order&High-order Feature Extractor,Prediction`

- Input&Embedding
>  The  data in CTR estimation task  usually includes high sparse,high cardinality 
  categorical features  and some dense numerical features.  

>  Since DNN are good at handling dense numerical features,we usually map the sparse categorical 
  features to dense numerical through `embedding technique`.  

 > For numerical features,we usually apply `discretization` or `normalization` on them.

- Feature Extractor
 > Low-order Extractor learns feature interaction through  product between vectors.Factorization-Machine and it's variants are widely used to learn the low-order feature interaction.

 > High-order Extractor learns feature combination through complex neural network functions like MLP,Cross Net,etc.

## Models


### PNN (Product-based Neural Network)

PNN concatenates sparse feature embeddings and the product between  embedding vectors as the input of MLP. 

[**PNN Model API**](./deepctr_torch.models.pnn.html)

![PNN](../pics/PNN.png)

[Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)


### Wide & Deep

WDL's deep part concatenates sparse feature embeddings as the input of MLP,the wide part use handcrafted feature as input.
The logits of deep part and wide part are added to get the prediction probability.

[**WDL Model API**](./deepctr_torch.models.wdl.html)

![WDL](../pics/WDL.png)

[Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.](https://arxiv.org/pdf/1606.07792.pdf)


### DeepFM

DeepFM can be seen as an improvement of WDL and FNN.Compared with WDL,DeepFM use
FM instead of LR in the wide part and use concatenation of embedding vectors as the input of MLP in the deep part.
Compared with FNN,the embedding vector of FM and input to MLP are same.
And they do not need a FM pretrained vector to initialiaze,they are learned end2end. 

[**DeepFM Model API**](./deepctr_torch.models.deepfm.html)

![DeepFM](../pics/DeepFM.png)

[Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.](http://www.ijcai.org/proceedings/2017/0239.pdf)

### MLR(Mixed Logistic Regression/Piece-wise Linear Model)

MLR can be viewed as a combination of $2m$ LR model, $m$  is the piece(region) number.
$m$ LR model learns the weight that the sample belong to each region,another m LR model learn sample's click probability in the region.
Finally,the sample's CTR is a weighted sum of each region's click probability.Notice the weight is normalized weight.

[**MLR Model API**](./deepctr_torch.models.mlr.html)

![MLR](../pics/MLR.png)

[Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.](http://arxiv.org/abs/1704.05194)


### NFM (Neural Factorization Machine)

NFM use a bi-interaction pooling layer to learn feature interaction between
embedding vectors and compress the result into a singe vector which has the same size as a single embedding vector.
And then fed it into a MLP.The output logit of MLP and the output logit of linear part are added to get the prediction probability. 

[**NFM Model API**](./deepctr_torch.models.nfm.html)

![NFM](../pics/NFM.png)

[He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](https://arxiv.org/pdf/1708.05027.pdf)


### AFM (Attentional Factorization Machine)

AFM is a variant of FM,tradional FM sums the inner product of embedding vector uniformly.
AFM can be seen as weighted sum of feature interactions.The weight is learned by a small MLP. 

[**AFM Model API**](./deepctr_torch.models.afm.html)

![AFM](../pics/AFM.png)

[Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.](http://www.ijcai.org/proceedings/2017/435)


### DCN (Deep & Cross Network)

DCN use a Cross Net to learn both low and high order feature interaction explicitly,and use a MLP to learn feature interaction implicitly.
The output of Cross Net and MLP are concatenated.The concatenated vector are feed into one fully connected layer to get the prediction probability. 

[**DCN Model API**](./deepctr_torch.models.dcn.html)

![DCN](../pics/DCN.png)

[Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123) 


### xDeepFM

xDeepFM use a Compressed Interaction Network (CIN) to learn both low and high order feature interaction explicitly,and use a MLP to learn feature interaction implicitly.
In each layer of CIN,first compute outer products between $x^k$ and $x_0$ to get a tensor $Z_{k+1}$,then use a 1DConv to learn feature maps $H_{k+1}$ on this tensor.
Finally,apply sum pooling on all the feature maps $H_k$ to get one vector.The vector is used to compute the logit that CIN contributes.

[**xDeepFM Model API**](./deepctr_torch.models.xdeepfm.html)

![CIN](../pics/CIN.png)

![xDeepFM](../pics/xDeepFM.png)


[Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.](https://arxiv.org/pdf/1803.05170.pdf)

### AutoInt(Automatic Feature Interaction)

AutoInt use a interacting layer to model the interactions between different features.
Within each interacting layer, each feature is allowed to interact with all the other features and is able to automatically identify relevant features to form meaningful higher-order features via the multi-head attention mechanism.
By stacking multiple interacting layers,AutoInt is able to model different orders of feature interactions. 

[**AutoInt Model API**](./deepctr_torch.models.autoint.html)

![InteractingLayer](../pics/InteractingLayer.png)

![AutoInt](../pics/AutoInt.png)

[Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)

### ONN(Operation-aware Neural Networks for User Response Prediction)

ONN models second order feature interactions like like FFM and preserves second-order interaction information  as much as possible.Further more,deep neural network is used to learn higher-ordered feature interactions. 

[**ONN Model API**](./deepctr_torch.models.onn.html)

![ONN](../pics/ONN.png)

[Yang Y, Xu B, Shen F, et al. Operation-aware Neural Networks for User Response Prediction[J]. arXiv preprint arXiv:1904.12579, 2019.](https://arxiv.org/pdf/1904.12579.pdf)

### FiBiNET(Feature Importance and Bilinear feature Interaction NETwork)

Feature Importance and Bilinear feature Interaction NETwork is proposed to dynamically learn the feature importance and fine-grained feature interactions. On the one hand, the FiBiNET can dynamically learn the importance of fea- tures via the Squeeze-Excitation network (SENET) mechanism; on the other hand, it is able to effectively learn the feature interactions via bilinear function.

[**FiBiNET Model API**](./deepctr_torch.models.fibinet.html)  

![FiBiNET](../pics/FiBiNET.png)

[Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.](https://arxiv.org/pdf/1905.09433.pdf)


## Layers

The models of deepctr are modular,
so you can use different modules to build your own models.

You can see layers API in [Layers](./Layers.html) 
