from glob import iglob
from os.path import join
import numba
import numpy as np
import pandas as pd


def read_multiples_csv(path, fn_regex=r'*.csv'):
    return pd.concat((pd.read_csv(f) for f in iglob(
        join(path, '**', fn_regex), recursive=True)), ignore_index=True)


def _keep(window, windows):
    """Helper function for creating rolling windows."""
    windows.append(window.copy())
    return -1.  # Float return value required for Pandas apply.


def create_rolling_features_label(series, window_size, pred_offset, pred_n=1):
    """Computes rolling window of the series and creates rolling window of label.
    Args:
      series: A Pandas Series. The indices are datetimes and the values are
        numeric type.
      window_size: integer; steps of historical data to use for features.
      pred_offset: integer; steps into the future for prediction.
      pred_n: integer; window size of label.
    Returns:
      Pandas dataframe where the index is the datetime predicting at. The columns
      beginning with "-" indicate windows N steps before the prediction time.
    Examples:
      >>> series = pd.Series(np.random.random(6),index=pd.date_range(start='1/1/2018', end='1/06/2018'))
      # Example #1:
      >>> series
      2018-01-01    0.803948
      2018-01-02    0.269849
      2018-01-03    0.971984
      2018-01-04    0.809718
      2018-01-05    0.324454
      2018-01-06    0.229447
      >>> window_size = 3 # get 3 months of historical data
      >>> pred_offset = 1 # predict starting next month
      >>> pred_n = 1 # for predicting a single month
      >>> utils.create_rolling_features_label(series,
                                              window_size,
                                              pred_offset,
                                              pred_n)
      pred_datetime -3_steps      -2_steps        -1_steps        label
      2018-01-04    0.803948      0.269849        0.971984        0.809718
      2018-01-05    0.269849      0.971984        0.809718        0.324454
      2018-01-06    0.971984      0.809718        0.324454        0.229447
      # Example #2:
      >>> window_size = 3 # get 3 months of historical data
      >>> pred_offset = 2 # predict starting 2 months into future
      >>> pred_n = 1 # for predicting a single month
      >>> utils.create_rolling_features_label(series,
                                              window_size,
                                              pred_offset,
                                              pred_n)
      pred_datetime       -4_steps        -3_steps        -2_steps        label
      2018-01-05    0.803948      0.269849        0.971984        0.324454
      2018-01-06    0.269849      0.971984        0.809718        0.229447
      # Example #3:
      >>> window_size = 3 # get 3 months of historical data
      >>> pred_offset = 1 # predict starting next month
      >>> pred_n = 2 # for predicting a multiple months
      >>> utils.create_rolling_features_label(series,
                                              window_size,
                                              pred_offset,
                                              pred_n)
      pred_datetime -3_steps      -2_steps        -1_steps        label_0_steps
      label_1_steps
      2018-01-04    0.803948      0.269849        0.971984        0.809718
      0.324454
      2018-01-05    0.269849      0.971984        0.809718        0.324454
      0.229447
    """
    if series.isnull().sum() > 0:
        raise ValueError('Series must not contain missing values.')
    if pred_n < 1:
        raise ValueError('pred_n must not be < 1.')
    if len(series) < (window_size + pred_offset + pred_n):
        raise ValueError('window_size + pred_offset + pred_n must not be greater '
                         'than series length.')
    total_steps = len(series)

    def compute_rolling_window(series, window_size):
        # Accumulate series into list.
        windows = []
        series.rolling(window_size) \
            .apply(_keep, args=(windows,))
        return np.array(windows)

    features_start = 0
    features_end = total_steps - (pred_offset - 1) - pred_n
    historical_windows = compute_rolling_window(
        series[features_start:features_end], window_size)
    # Get label pred_offset steps into the future.
    label_start, label_end = window_size + pred_offset - 1, total_steps
    label_series = series[label_start:label_end]
    y = compute_rolling_window(label_series, pred_n)
    if pred_n == 1:
        # TODO(crawles): remove this if statement/label name. It's for backwards
        # compatibility.
        columns = ['label']
    else:
        columns = ['label_{}_steps'.format(i) for i in range(pred_n)]
    # Make dataframe. Combine features and labels.
    label_ix = label_series.index[0:len(label_series) + 1 - pred_n]
    df = pd.DataFrame(y, columns=columns, index=label_ix)
    df.index.name = 'pred_date'
    # Populate dataframe with past sales.
    for day in range(window_size - 1, -1, -1):
        day_rel_label = pred_offset + window_size - day - 1
        df.insert(0, '-{}_steps'.format(day_rel_label), historical_windows[:, day])
    return df


def sigma_filter(df, tolerance=3):
    ''' sigma clipping '''
    df['meter_reading_ln'] = np.log1p(df.meter_reading)
    stats = df.reset_index().set_index('timestamp').groupby(['building_id', 'meter']) \
        .rolling(24 * 7, min_periods=2, center=True).meter_reading_ln.agg(['median'])
    std = df.reset_index().set_index('timestamp').groupby(['building_id', 'meter']).meter_reading_ln.std()
    stats['max'] = np.expm1(stats['median'] + tolerance * std)
    stats['min'] = np.expm1(stats['median'] - tolerance * std)
    stats['median'] = np.expm1(stats['median'])
    df = df.merge(stats[['median', 'min', 'max']], left_on=['building_id', 'meter', 'timestamp'], right_index=True)
    return df[(df.meter_reading <= df['max']) & git(df.meter_reading >= df['min'])]


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
