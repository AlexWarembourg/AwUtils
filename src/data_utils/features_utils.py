import numba
import numpy as np


@numba.jit(nopython=True)
def find_start_end(data: np.ndarray):
    """
    Calculates start and end of real traffic data. Start is an index of first non-zero, non-NaN value,
     end is index of last non-zero, non-NaN value
    :param data: Time series, shape [n_sku, n_days]
    :return:
    """
    n_sku = data.shape[0]
    n_days = data.shape[1]
    start_idx = np.full(n_sku, -1, dtype=np.int32)
    end_idx = np.full(n_sku, -1, dtype=np.int32)
    for sku in range(n_sku):
        # scan from start to the end
        for day in range(n_days):
            if not np.isnan(data[sku, day]) and data[sku, day] > 0:
                start_idx[sku] = day
                break
        # reverse scan, from end to start
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[sku, day]) and data[sku, day] > 0:
                end_idx[sku] = day
                break
    return start_idx, end_idx


@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    if series.shape[0] > lag:
        s1 = series[lag:]
        s2 = series[:-lag]
        ms1 = np.mean(s1)
        ms2 = np.mean(s2)
        ds1 = s1 - ms1
        ds2 = s2 - ms2
        divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
        return np.sum(ds1 * ds2) / divider if divider != 0.0 else 0.0
    else:
        return 0.0


def batch_autocorr(data, lag, threshold=1, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_sku, n_days]
    :param lag: Autocorrelation lag
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    starts, ends = find_start_end(data)
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len / lag if lag != 0 else real_len
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag - 1)
            c_366 = single_autocorr(series, lag + 1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    return corr


def get_map(df, col):
    uniques = df[col].unique()
    mapping_dict = {}
    for i in range(len(uniques)):
        mapping_dict[uniques[i]] = i
    df[col] = df[col].map(mapping_dict)
    return df, len(uniques)
