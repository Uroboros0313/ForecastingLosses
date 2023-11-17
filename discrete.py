import torch as th



class TweedieLoss():
    pass


class PossionLogLikelihoodLoss():
    def __init__(self):
        pass

    def __call__(self, z, mu):
        return self.__loss_fn__(z, mu)

    def __loss_fn__(self, z, mu):
        loss = mu - z * th.log(mu)
        return th.sum(loss)


class NegativeBinomialLogLikelihoodLoss():
    def __init__(self):
        pass

    def __call__(self, z, mu, alpha):
        return self.__loss_fn__(z, mu, alpha)

    def __loss_fn__(self, z, mu, alpha):
        loss = th.lgamma(z + 1.0 / alpha) - th.lgamma(z + 1.0) - th.lgamma(1.0 / alpha) \
            - 1.0 / alpha * th.log(1 + alpha * mu) + z * th.log(alpha * mu / (1 + alpha * mu))

        return th.sum(loss)
    