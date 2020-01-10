from glob import iglob
from os.path import join

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
