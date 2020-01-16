from glob import iglob
from os.path import join
import numba
import numpy as np
import pandas as pd
from src.data_utils.holidays import UACalendar


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


def expend_event(df, event_name, direction=None, n_ahead=None, n_previous=None):
    """
    temporaly expand a bolean along an axis.
    Args:
        df: dataframe
        event_name: string : name of columns
        direction: string type : both or None
        n_ahead: int : numbers of day ahead to set at 1
        n_previous: int : numbers of day before to set at 1

    Returns: pd.Series
    exemple:
        improved_calendar["orthodox_xmas_flg"] = np.where(improved_calendar["Holiday_Name"] == 'Orthodox Christmas', 1, 0)
        improved_calendar["orthodox_xmas_flg"] = expend_event(improved_calendar, "orthodox_xmas_flg", direction="both",
                                                              n_ahead=3,
                                                              n_previous=6)
    """
    # get the right shape to fill the container zero matrix
    right_shape = int(n_ahead + n_previous) if direction == "both" else n_ahead if n_ahead is not None else n_previous
    # init a container that will be willed with shifted series
    container = np.zeros((df.shape[0], right_shape))
    if direction == "both":
        for r in range(-n_previous, n_ahead):
            # fill an array with corresponding
            container[:, r] = np.roll(df[event_name].values, r)
    else:
        for r in range(right_shape):
            container[:, r] = np.roll(df[event_name].values, r)
    # sum up along one axis 1 to 1D array
    # since we sum several array we need to clip it at 1 to get bolean
    container = np.clip(np.sum(container, axis=1), a_min=0, a_max=1)
    return container


def _count_holidays(dates, weeks):
    """Count number of holidays spanned in prediction windows."""
    cal = UACalendar()
    holidays = cal.holidays(start=dates.min(), end=dates.max())

    def count_holidays_during_weeks(date):
        n_holidays = 0
        beg = date
        end = date + pd.DateOffset(weeks=weeks)  # to keep  week format we just put month to 0
        for h in holidays:
            if beg <= h < end:
                n_holidays += 1
        return n_holidays

    return pd.Series(dates).apply(count_holidays_during_weeks)


def _get_day_of_month(x):
    """From a datetime object, extract day of month."""
    return int(x.strftime('%d'))


def add_date_features(df, dates, weeks, inplace=False):
    """Create features using date that is being predicted on."""
    if not inplace:
        df = df.copy().sort_values(dates.name)
    df["dow"] = dates.dt.dayofweek
    df['dom'] = dates.dt.day
    df['doy'] = dates.dt.dayofyear
    df["is_wknd"] = np.where(dates.dt.dayofyear.isin([5, 6]), 1, 0)
    df['month'] = dates.dt.month
    df["month_split"] = dates.dt.day % 6  # reset cum_count every 6 day
    df["week"] = dates.dt.week
    df["woy"] = dates.dt.weekofyear
    df["quarter"] = dates.dt.quarter
    df["season"] = dates.apply(lambda dt: (dt.month % 12 + 3) // 3)
    df['year'] = dates.dt.year
    df["doy_cos"] = np.cos(dates.dt.dayofyear)
    df["doy_sin"] = np.sin(dates.dt.dayofyear)
    df["dow_cos"] = np.cos(dates.dt.dayofweek)
    df["dow_sin"] = np.sin(dates.dt.dayofweek)
    df["flg_global_holiday"] = np.where(df['Holiday_Name'].notnull(), 1, 0)
    df['n_holidays'] = _count_holidays(pd.DatetimeIndex(dates), weeks).values
    df["next_day_is_holiday"] = df["flg_global_holiday"].shift(-1)
    df["prev_day_is_holiday"] = df["flg_global_holiday"].shift(1)
    return df


class TimeSeries():
    """Time Series modelling based on the initial Uber article. The first part
    is about the features building, and then the modelling itself. Note that
    the method to create models are almost externals at the class, so they are
    called here but the user needs to manually input the parameters (the number
    of hidden layers, length of timesteps ...)

    Attributes
    -----------------
    - window: rolling window of the time series used to create variables

    Usage
    - compute_time_series_features: computes the time series features
    - sequential_autoencoder: autoencoder
    - create_variable_for_model: create the variable to be inputed in the
    recurrent neural network
    - forecast_model: create LSTM model

    """

    def __init__(self, window):
        """Initialize the algorithm, with a fixed rolling window"""
        self.window = window

    def _extract_time_series(self, x, i, taille):
        """Function extracting the sub time series from a pandas Series. We use
        this function  because the pandas Series does not have same properties
        as numpy array"""

        # Need to differentiate the last case
        if i != taille:
            x_int = x[x.index[i - self.window]:x.index[i - 1]]
        else:
            x_int = x[x.index[i - self.window]:]

        return x_int

    def ts_mean(self, x):
        """Computes the time series mean for a rolling window"""

        # Change name
        x.name = 'average'

        return x.rolling(self.window).mean()

    def ts_var(self, x):
        """Computes the time series variance for a rolling window"""

        # Change name
        x.name = 'variance'

        return x.rolling(self.window).var()

    def ts_autocorrelation(self, x):
        """Computes the time series autocorrelation for a rolling window"""

        # Initialize output
        res = pd.Series(data=None, index=x.index, name='autocorr')

        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            res[res.index[i - 1]] = self._extract_time_series(x, i, x.shape[0]).autocorr()

        return res

    def ts_entropy(self, x):
        """Computes the time series entropy for a rolling window"""

        # Initialize output
        res = pd.Series(data=None, index=x.index, name='entropy')

        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            res[res.index[i - 1]] = entropy(self._extract_time_series(x, i, x.shape[0]))

        return res

    def ts_trend(self, x):
        """Computes the trend coefficient and the variance of the residuals of
        the time series versus the trend, for a rolling window"""

        # Initialize outputs
        res1 = pd.Series(data=None, index=x.index, name='trend')
        res2 = pd.Series(data=None, index=x.index, name='trend_p_value')
        res3 = pd.Series(data=None, index=x.index, name='residuals_variance')

        # Create the X, to compute the trend
        x_int = np.arange(self.window)
        x_int = sm.add_constant(x_int)

        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            # Extract the corresponding time series window
            y_int = self._extract_time_series(x, i, x.shape[0])

            # Linear Model
            m = sm.OLS(y_int, x_int)
            results = m.fit()

            # Trend value and its p-value
            res1[res1.index[i - 1]] = results.params['x1']
            res2[res2.index[i - 1]] = results.pvalues['x1']

            # Strength of trend
            res3[res3.index[i - 1]] = np.var(results.predict(x_int) - y_int)

        return res1, res2, res3

    def ts_spike(self, x):
        """Computes the spike of a time series, for a rolling window"""

        # Initialize output
        res = pd.Series(data=None, index=x.index, name='spike')

        # Loop through the time series
        for i in range(self.window, x.shape[0] + 1):
            # Extract the corresponding time series window and computes the spike
            x_int = self._extract_time_series(x, i, x.shape[0])
            res[res.index[i - 1]] = self.function_spike(x_int)

        return res

    def function_spike(self, x):
        """Computes the spike function for a time series"""

        # Computes the max and min spikes versus the average
        M, m = x.max() - x.mean(), x.min() - x.mean()

        # Return the biggest spike
        if np.abs(M) > np.abs(m):
            xtrm = M
        else:
            xtrm = m

        return xtrm

    def ts_crossing_points(self, x):
        """Computes the number of times the mean is crossed by the time series
        for a rolling window"""

        # Initialize output
        res = pd.Series(data=None, index=x.index, name='crossing_points')

        # Loop through time series
        for i in range(self.window, x.shape[0] + 1):
            # Extract the corresponding time series window and computes the crossing points
            x_int = self._extract_time_series(x, i, x.shape[0])
            res[res.index[i - 1]] = self.function_crossing_points(x_int)

        return res

    def function_crossing_points(self, x):
        """Computes the number of times the times series cross its mean"""

        # Reduce the mean
        x = np.sign(x - x.mean())

        # Number of times the mean is crossed
        x = x.diff().fillna(0)
        return np.abs(x).sum() / 2

    def compute_time_series_features(self, x):
        """Computes the different features of the time series"""

        # In case, transform the initial input into a pandas Series
        x = pd.Series(x, name='time_series')

        # Computes the individual features
        x1 = self.ts_mean(x)
        x2 = self.ts_var(x)
        x3 = self.ts_autocorrelation(x)
        x4 = self.ts_entropy(x)
        x5, x6, x7 = self.ts_trend(x)
        x8 = self.ts_spike(x)
        x9 = self.ts_crossing_points(x)

        # Concatenate them into a DataFrame
        res = pd.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9], axis=1)

        # Look also at the features changes
        res_diff = res.diff()
        res_diff.columns = [x + '_diff' for x in res_diff.columns]

        return pd.concat([res, res_diff], axis=1)

    def sequential_autoencoder(self, x, num_layers, timesteps=5):
        """LSTM Autoencoder to extract high level features
        Note that the x input is still in a raw format [m,n] where m are dates
        and n are the features. The sequence length is still given by timesteps,
        and some data manipulation needs to be made before inputing the data in
        the model. This is why there is the function create_variable_for_model
        """

        # Input
        inputs = Input(shape=(timesteps, x.shape[1]))

        # Encoder
        encoder = LSTM(num_layers, activation='tanh')(inputs)

        # Decoder
        decoder = RepeatVector(timesteps)(encoder)
        decoder = LSTM(x.shape[1], return_sequences=True, activation='tanh')(decoder)

        # Models
        sequential_autoencoder = Model(inputs, decoder)
        sequential_autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        encoder = Model(inputs, encoder)
        encoder.compile(loss='mean_squared_error', optimizer='adam')

        return sequential_autoencoder, encoder

    def forecast_model(self, x, num_layers, timesteps=5):
        """LSTM model for forecasting.
        Note that the x input is still in a raw format [m,n] where m are dates
        and n are the features. The sequence length is still given by timesteps,
        and some data manipulation needs to be made before inputing the data in
        the model. This is why there is the function create_variable_for_model
        """

        # LSTM Model
        model = Sequential()
        model.add(LSTM(units=num_layers, input_shape=(timesteps, x.shape[1]),
                       activation='tanh'))
        model.add(Dense(1, activation=None))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def create_variable_for_model(self, x, timesteps=5):
        """Transform the variables inputs into an appropriate format for the
        second model. The x input needs to be already in a numpy format."""

        # Initialize output
        res = []

        # Create the output
        for idx in range(x.shape[0] - timesteps):
            res.append(x[idx:idx + timesteps, :])

        return np.array(res)
