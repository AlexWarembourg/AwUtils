import numpy as np


def mask_true_zero(y_true, y_hat):
    if not isinstance(y_true, np.ndarray) and not isinstance(y_hat, np.ndarray):
        y_true, y_hat = np.array(y_true), np.array(y_hat)
        assert y_true.shape[0] == y_hat.shape[
            0], f"Array length mismatch with y_true : {y_true.shape[0]}  =/= y_hat {y_hat.shape[0]}"
    stack = np.vstack((y_true, y_hat)).T
    inds = np.where(stack[:, 0] > 0)
    stack_non_zero = stack[inds]
    y_true, y_hat = stack_non_zero[:, 0], stack_non_zero[:, 1]
    return y_true, y_hat


def calculate_stats(y_true, predictions):
    """
    Basics stats ..
    """
    max_true, max_prev = np.max(y_true), np.max(predictions)
    mu, sigma = np.mean(y_true), np.std(y_true)
    mu_hat, sigma_hat = np.mean(predictions), np.std(predictions)
    return max_true, max_prev, mu, sigma, mu_hat, sigma_hat


def calculate_rmse(residuals):
    """Root mean squared error.
    N.B : as a cost function we predict the average demand
    """
    return np.sqrt(np.mean(np.square(residuals)))


def calculate_percent_mae(y_true, residuals):
    """
    Percent Mean absolute error.
    This is a mesure of over/under estimation
    warning : you will always get a biased forecast if the demand mean differs from the median.
    N.B : as a cost function with predict the median demand
     """
    return np.mean(np.abs(residuals)) / np.mean(y_true)


def calculate_fiability(y_true, predictions):
    return np.clip(1 - abs(y_true - predictions) / y_true, 0, 1)


def calculate_pondered_fiability(y_true, predictions):
    """
    Pondered reliability is a pondered mesure adequat to focus on high qty
    Parameters
    """
    fiab_ = calculate_fiability(y_true, predictions)
    fiab_ = np.where(y_true == predictions, 1, fiab_)
    return np.sum(y_true * fiab_) / np.sum(y_true)


def calculate_outlier_sensitivity(y_true):
    """
    closer to 1 more sensitive is it
    """
    return (np.mean(y_true) - np.median(y_true)) / np.mean(y_true)


def calculate_smape(y_true, predictions):
    """ symmetric mean average percent error """
    return 100 / y_true.shape[0] * np.sum(2 * np.abs(predictions - y_true) / (np.abs(y_true) + np.abs(predictions)))


def report(name=None):
    if name is not None:
        print_string = '{} results'.format(name)
        print(print_string)
        print('~' * len(print_string))
    print('True max : {:2.2f} | Pred max {:2.2f}'.format(max_true, max_prev))
    print('True mean : {:2.2f} | Pred mean : {:2.2f}'.format(mu, pred_mu))
    print('True std : {:2.3f} | Pred std : {:2.3f}'.format(sigma, pred_sigma))
    print('RMSE: {:2.3f}\nMAE: {:2.3f}\nSMAPE: {:2.3f}\nOUTLIER_SENSIVITY : {:2.3f}\nPONDERED RELIABILITY : {:2.3f}' \
          .format(rmse, mae, smape, outlier_sens, pond_reliability))
