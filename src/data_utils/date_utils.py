import numpy as np
import pandas as pd


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
    df["is_wknd"] = np.where(df["doy"].isin([5, 6]), 1, 0)
    df['dom'] = pd.DatetimeIndex(dates).map(_get_day_of_month)
    df['doy'] = dates.dt.dayofyear
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
