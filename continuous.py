from collections import Any, Iterable

import torch as th


class HuberLoss():
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, preds, labels):
        return self.__loss_fn__(preds, labels)

    def __loss_fn__(self, preds, labels):
        abs_diff = th.abs(preds - labels)
        mask_ = (abs_diff > self.delta)

        mse = 0.5 * (preds - labels) ** 2
        mae = self.delta * th.abs(preds - labels) - 0.5 * self.delta ** 2

        loss = th.sum(mse[~mask_]) + th.sum(mae[mask_])
        return loss


class LogCoshLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, preds, labels):
        return self.__loss_fn__(preds, labels)

    def __loss_fn__(self, preds, labels):
        loss = th.log(th.cosh(preds - labels))
        return loss


class SmoothL1Loss():
    def __init__(self, beta=0.1):
        self.beta = beta

    def __call__(self, preds, labels):
        return self.__loss_fn__(preds, labels)

    def __loss_fn__(self, preds, labels):
        abs_diff = th.abs(preds - labels)
        mask_ = (abs_diff > self.delta)

        mse = 0.5 * (preds - labels) ** 2 / self.beta
        mae = th.abs(preds - labels) - 0.5 * self.beta

        loss = th.sum(mse[~mask_]) + th.sum(mae[mask_])
        return loss


class QuantileLoss():
    def __init__(self, quantile):
        assert isinstance(quantile, (float, Iterable))

        self.quantile = quantile

    def __call__(self, preds, labels):
        if isinstance(self.quantile, float):
            return self.__loss_fn__(preds, labels, self.quantile)
        else:
            quantile_loss = 0.0
            for q in self.quantiles:
                quantile_loss += self.__loss_fn__(preds, labels, q)
            return quantile_loss

    def __loss_fn__(self, preds, labels, q):
        loss = q * max(0, labels - preds) + (1 - q) * max(0, preds - labels)
        return th.sum(loss)


class GaussianNLLLoss():
    def __init__(self, eps=1e-4):
        self.eps = eps

    def __call__(self, z, mu, sigma):
        return self.__loss_fn__(z, mu, sigma)

    def __loss_fn__(self, z, mu, sigma):
        loss = th.log(th.max(sigma ** 2, self.eps)) + (z - mu) ** 2 / (sigma ** 2)
        return th.sum(loss)
