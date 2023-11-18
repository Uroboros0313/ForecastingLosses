import torch as th


class TweedieNLLLoss():
    pass


class PossionNLLLoss():
    def __init__(self):
        pass

    def __call__(self, z, mu):
        return self.__loss_fn__(z, mu)

    def __loss_fn__(self, z, mu):
        '''
        The approximation is used for target values more than 1. 
        For targets less or equal to 1, zeros are added to the loss.
        '''
        mask_ = (z <= 1.0)
        possion_nll = mu - z * th.log(mu)
        loss = possion_nll[mask_]

        return th.sum(loss)


class NegativeBinomialNLLLoss():
    def __init__(self):
        pass

    def __call__(self, z, mu, alpha):
        return self.__loss_fn__(z, mu, alpha)

    def __loss_fn__(self, z, mu, alpha):
        loss = th.lgamma(z + 1.0 / alpha) - th.lgamma(z + 1.0) - th.lgamma(1.0 / alpha) \
               - 1.0 / alpha * th.log(1 + alpha * mu) + z * th.log(alpha * mu / (1 + alpha * mu))

        return th.sum(loss)
