import numpy as np


def gaussian_kl(mu, log_variance, prior_mu, prior_sigma):
    """KL(q || N(prior_mu, prior_sigma**2 I)), summed over latent dimensions."""
    variance = log_variance.exp()

    return (
        0.5
        * (
            (variance + (mu - prior_mu).square()) / prior_sigma**2
            - 1
            + 2 * np.log(prior_sigma)
            - log_variance
        )
    ).sum(-1)
