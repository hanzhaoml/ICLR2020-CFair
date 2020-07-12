# Conditional Learning of Fair Representations

PyTorch code for the paper [Conditional Learning of Fair Representations](https://openreview.net/forum?id=Hkekl0NFPr) by [Han Zhao](http://www.cs.cmu.edu/~hzhao1/), [Amanda Coston](http://www.cs.cmu.edu/~acoston/), [Tameem Adel](https://tameemadel.wordpress.com/), and [Geoff Gordon](http://www.cs.cmu.edu/~ggordon/). 


## Summary

CFair is an algorithm for learning fair representations that aims to reduce the equalized odds gap as well as accuracy disparity simultaneously under the classification setting. For a classification problem with $K$ output classes and a binary protected attribute, e.g., race or gender, the overall model contains the following three parts:
*   A feature transformation, parametrized by a neural network.
*   A classifier for the target task of interest, e.g., income prediction or recidivism prediction. 
*   A set of $K$ adversarial classifiers (auditors for the protected attribute), one for each output class, trying to align the conditional distributions of representations across different groups. The $k$-th $(k\in [K])$ adversary corresponds to a binary classifier that tries to discriminate between instances from different groups with class label $k$.

Note that to minimize the loss of target prediction, we use the `Balanced Error Rate`, which is implemented as a class-reweighted version of the `Cross Entropy Loss`, and the reweighting is given by $1 / \Pr(Y = k), k\in[K]$.
More detailed description about the practical implementation of the objective function could be found in Section 3.4 of the paper [Conditional Learning of Fair Representations](https://openreview.net/forum?id=Hkekl0NFPr). 

## Optimization

It is notoriously difficulty to optimize a minimax problem when it is nonconvex and noncave. Our goal is to converge to a saddle point. In this code repo we use the gradient descent-ascent algorithm to optimize the objective function. Intuitively, this means that we use simultaneous gradient updates for all the components in the model. As a comparison, in block coordinate method, we would either fix the set of $K$ adversaries or the feature extractor and the hypothesis, and optimize the other until convergence, and then iterate from there. 

Specifically, we use the following gradient reversal layer to implement this method. Code snippet in PyTorch shown as follows:

```python
class GradReverse(Function):
    """
    Implement the gradient reversal layer for gradient descent-ascent algorithm.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)
```

## Prerequisites

*   `Python 3.6.6`
*   `PyTorch >= 1.0.0`
*   `Numpy`

__________

This part explains how to reproduce the experiments on `Adult` and `COMPAS` datasets in the paper. 

### Training + Evaluation

For the `Adult` dataset, run
```python
python main_adult.py -m [mlp|fair|cfair|cfair-eo]
```
The code implements 4 different algorithms:
*   `mlp`: NoDebias
*   `fair`: Fair (to approximately achieve demographic parity)
*   `cfair-eo`: CFair-EO (to approximately achieve equalized odds)
*   `cfair`: CFair (to approximately achieve both equalized odds and accuracy parity)

Note that there are other hyperparameters could be set by using different options provided by `argparse`:
*   Coefficient for the adversarial loss: `-u` or `--mu`. 
*   Number of training epoch: `-e` or `--epoch`.
*   Initial learning rate: `-r` or `--lr`.
*   Batch size for SGD-style algorithm: `-b` or `--batch_size`.

Similarly, for the `COMPAS` dataset, run:
```python
python main_compas.py -m [mlp|fair|cfair|cfair-eo]
```

## Citation
If you use this code for your research and find it helpful, please cite our paper [Conditional Learning of Fair Representations](https://openreview.net/forum?id=Hkekl0NFPr):
```
@inproceedings{zhao2019conditional,
  title={Conditional Learning of Fair Representations},
  author={Zhao, Han and Coston, Amanda and Adel, Tameem and Gordon, Geoffrey J},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## Contact
Please email to [han.zhao@cs.cmu.edu](mailto:han.zhao@cs.cmu.edu) should you have any questions or comments.
