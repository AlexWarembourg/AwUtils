import gc
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from MultiDimensionalDataFrame import DataFrame


def get_stacked_data(multi_idx, dynamic, static, data_dir, date_range):
    data = []
    data_cols = dynamic + static
    for data_set in data_cols:
        if data_set in dynamic and data_set == "target_df":
            data.append(pd.DataFrame(
                np.log1p(np.load(os.path.join(data_dir, '{}.npy'.format(data_set)))),
                columns=date_range, index=multi_idx)
            )
        elif data_set in dynamic and data_set != "target_df":
            data.append(pd.DataFrame(
                np.load(os.path.join(data_dir, '{}.npy'.format(data_set))),
                columns=date_range, index=multi_idx)
            )
        else:
            data.append(pd.Series(np.load(os.path.join(data_dir, '{}.npy'.format(data_set)))))

    X = DataFrame(columns=data_cols, data=data)
    del data
    gc.collect()
    return X


def assert_shape_of_flat_array(static_data):
    """
    tricky way to assert that a flat_array is realy flat
    """
    return str(static_data.shape).split(',')[1] == ')'


def create_xy_span(df, pred_start, timesteps, is_train=True, n_day_ahead=16):
    # took 90 day before pred start and -1 before pred start
    X = df[pd.date_range(pred_start - timedelta(days=timesteps), pred_start - timedelta(days=1))].values
    if is_train:
        y = df[pd.date_range(pred_start, periods=n_day_ahead)].values
    else:
        y = None
    return X, y


def tabular_to_sequential(dataframe, timesteps, n_day_ahead, pred_start):
    # [len, timesteps] = (5000, 90)
    X, y = create_xy_span(dataframe["demand_df"], pred_start, timesteps, n_day_ahead=n_day_ahead)
    # is0 = (X == 0).astype('uint8').reshape(-1, timesteps, 1)

    # we known his feature so we took 90days before and 16 after pred
    # so we got [len, timestep + range_to_predict] = (5000, 90+42)
    promo = dataframe["promo_df"][
        pd.date_range(pred_start - timedelta(days=timesteps), periods=timesteps + n_day_ahead)].values

    holiday = dataframe["holiday_df"][
        pd.date_range(pred_start - timedelta(days=timesteps), periods=timesteps + n_day_ahead)].values

    exposition = dataframe["exposition_df"][
        pd.date_range(pred_start - timedelta(days=timesteps), periods=timesteps + n_day_ahead)].values

    # same here
    weekday = np.tile(
        [d.weekday() for d in pd.date_range(pred_start - timedelta(days=timesteps), periods=timesteps + n_day_ahead)],
        (X.shape[0], 1)
    )

    dom = np.tile(
        [d.day - 1 for d in pd.date_range(pred_start - timedelta(days=timesteps), periods=timesteps + n_day_ahead)],
        (X.shape[0], 1)
    )

    month = np.tile([d.month - 1 for d in pd.date_range(pred_start - timedelta(days=timesteps),
                                                        periods=timesteps + n_day_ahead)], (X.shape[0], 1))

    # extend static feature to length of series (need flat array) -- > shape =  (length of series, )
    assert assert_shape_of_flat_array(dataframe["sku_id"]), "your array is not flat"
    assert assert_shape_of_flat_array(dataframe["store_id"]), "your array is not flat"

    # same here we known cat in feature
    cat_features = np.tile(
        np.stack(
            [dataframe["sku_id"], dataframe["store_id"], dataframe["flag_loylaty"],
             dataframe["lv1"], dataframe["lv2"], dataframe["lv3"], dataframe["lv4"],
             dataframe["format_store"]],
            axis=1)[:, None, ], (1, timesteps + n_day_ahead, 1)
    )

    # unknown feature (target)
    X = X.reshape(-1, timesteps, 1)

    # known feature
    promo_x = promo.reshape(-1, timesteps + n_day_ahead, 1)
    holiday = holiday.reshape(-1, timesteps + n_day_ahead, 1)
    exposition = exposition.reshape(-1, timesteps + n_day_ahead, 1)

    # shift series at row level
    lagged_sequence = np.roll(X, 1)
    lagged_sequence[:, 0] = 0
    lagged_sequence.reshape(-1, timesteps, 1)
    # we must to return a tuple of ([features1, features2], target)
    # ex_tuples = ([X, promo_x], y)
    return ([X, lagged_sequence, holiday, exposition, promo_x, weekday,
             dom, month, cat_features], y)


def train_generator(df, batch_size, timesteps, n_day_ahead, pred_start):
    batch_gen = df.batch_generator(batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=True)
    for batch in batch_gen:
        yield tabular_to_sequential(batch, timesteps, n_day_ahead, pred_start)


def flatten_forecast(yhat, multi_idx, n_day_ahead, forecast_start):
    df_preds = (pd.DataFrame(
        np.expm1(yhat), index=multi_idx,
        columns=pd.date_range(forecast_start, periods=n_day_ahead))
                .stack()
                .to_frame("qty_forecast")
                .reset_index()
                .rename(columns={"level_3": "dt_ticket_sale"})
                )
    del yhat
    return df_preds


def random_shift_slice(mat, start_col, timesteps, shift_range):
    shift = np.random.randint(shift_range + 1, size=(mat.shape[0], 1))
    shift_window = np.tile(shift, (1, timesteps)) + np.tile(np.arange(start_col, start_col + timesteps),
                                                            (mat.shape[0], 1))
    rows = np.arange(mat.shape[0])
    rows = rows[:, None]
    columns = shift_window
    return mat[rows, columns]


def get_map(df, col):
    uniques = df[col].unique()
    mapping_dict = {}
    for i in range(len(uniques)):
        mapping_dict[uniques[i]] = i
    df[col] = df[col].map(mapping_dict)
    return df, len(uniques)


def get_temporal_stack(dataframe, key_col=None, group_by_key=None, with_bool=True, is_numeric=False):
    if with_bool and not isinstance(dataframe[key_col], bool):
        dataframe[key_col] = dataframe[key_col].astype(bool)
    elif with_bool == False and not isinstance(dataframe[key_col], (int, float)):
        df, n_uniques = get_map(dataframe, key_col)
    if is_numeric:
        safe_filler = 0.0
    else:
        safe_filler = False if with_bool else n_uniques + 1
    return (
        dataframe
            .set_index(group_by_key)[[key_col]]
            .unstack(level=-1)
            .fillna(safe_filler)  # to be sure that cat not exist
            .pipe(set_cols,
                  fn=lambda x: x.columns.get_level_values(1))  # apply func in method chaining on intermediary DataFrame
            .astype('int')
    )


def set_cols(df, fn=lambda x: x.columns.map('_'.join), cols=None):
    """
    Sets the column of the data frame to the passed column list.
    """
    if cols:
        df.columns = cols
    else:
        df.columns = fn(df)
    return df
