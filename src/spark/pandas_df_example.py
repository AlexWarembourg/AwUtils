from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType, sum, max, col, concat, lit
import argparse
import sys
import os

# global setup to work around with pandas udf
# ! sudo pip3 install pyarrow=0.14.1
# see answers here https://stackoverflow.com/questions/58458415/pandas-scalar-udf-failing-illegalargumentexception
os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

from fbprophet import Prophet
import pandas as pd
import numpy as np

# define an output schema
schema = StructType([
    StructField("store", StringType(), True),
    StructField("item", StringType(), True),
    StructField("ds", DateType(), True),
    StructField("yhat", DoubleType(), True)
])


# show spark version
# sc

def GetHolidays():
    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        'ds': pd.to_datetime(['2013-01-12', '2013-07-12', '2013-12-24',
                              '2014-01-12', '2014-07-12', '2014-07-19',
                              '2014-07-02', '2014-12-24', '2015-07-11', '2015-12-24',
                              '2016-07-17', '2016-07-24', '2016-07-07',
                              '2016-07-24', '2016-12-24', '2017-07-17', '2017-07-24',
                              '2017-07-07', '2017-12-24']),
        'lower_window': 0,
        'upper_window': 2}
    )
    superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2013-01-01', '2013-01-21', '2013-02-14', '2013-02-18',
                              '2013-05-27', '2013-07-04', '2013-09-02', '2013-10-14', '2013-11-11', '2013-11-28',
                              '2013-12-25', '2014-01-01', '2014-01-20', '2014-02-14', '2014-02-17',
                              '2014-05-26', '2014-07-04', '2014-09-01', '2014-10-13', '2014-11-11', '2014-11-27',
                              '2014-12-25', '2015-01-01', '2015-01-19', '2015-02-14', '2015-02-16',
                              '2015-05-25', '2015-07-03', '2015-09-07', '2015-10-12', '2015-11-11', '2015-11-26',
                              '2015-12-25', '2016-01-01', '2016-01-18', '2016-02-14', '2016-02-15',
                              '2016-05-30', '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24',
                              '2016-12-25', '2017-01-02', '2017-01-16', '2017-02-14', '2017-02-20',
                              '2017-05-29', '2017-07-04', '2017-09-04', '2017-10-09', '2017-11-10', '2017-11-23',
                              '2017-12-25', '2018-01-01', '2018-01-15', '2018-02-14', '2018-02-19'
                              ]),
        'lower_window': 0,
        'upper_window': 3,
    })

    holidays = pd.concat((playoffs, superbowls))
    return holidays


@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def fit_pandas_udf(df):
    """
    :param df: Dataframe (train + test data)
    rows have to be identified as train or test data using a col called 'train' as a boolean
    :return: predictions as defined in the output schema
    """

    def train_fitted_prophet(df, cutoff):
        # train
        ts_train = (df
                    .query('date <= @cutoff')
                    .rename(columns={'date': 'ds', 'sales': 'y'})
                    .sort_values('ds')
                    )
        # test
        ts_test = (df
                   .query('date > @cutoff')
                   .rename(columns={'date': 'ds', 'sales': 'y'})
                   .sort_values('ds')
                   .assign(ds=lambda x: pd.to_datetime(x["ds"]))
                   .drop('y', axis=1)
                   )

        print(ts_test.columns)
        # get holidays
        holidays = GetHolidays()
        # init model
        m = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    holidays=holidays)
        m.fit(ts_train)

        # to date
        df["date"] = pd.to_datetime(df["date"])
        # at this step we predict the future and we get plenty of additional columns be cautious
        ts_hat = (m.predict(ts_test)[["ds", "yhat"]]
                  .assign(ds=lambda x: pd.to_datetime(x["ds"]))
                  ).merge(ts_test, on=["ds"], how="left")  # merge to retrieve item and store index
        # debug
        # print(ts_hat)
        return pd.DataFrame(ts_hat, columns=schema.fieldNames())

    return train_fitted_prophet(df, cutoff)


if __name__ == '__main__':
    spark = (SparkSession
             .builder
             .appName("forecasting")
             .config('spark.sql.execution.arrow.enable', 'true')
             .getOrCreate()
             )

    # read input data from :https://www.kaggle.com/c/demand-forecasting-kernels-only/data
    data_train = (spark
                  .read
                  .format("csv")
                  .option('header', 'true')
                  .load('Downloads/train.csv')
                  )

    data_test = (spark
                 .read
                 .format("csv")
                 .option('header', 'true')
                 .load('Downloads/test.csv')
                 .drop('id')
                 )
    # max train date
    cutoff = data_train.select(max(col('date'))).collect()[0][0]
    # add sales none col to match with union
    data_test = data_test.withColumn('sales', lit(None))
    # concat train test
    df = (data_train.union(data_test)).sort(col('date'))
    # fit
    global_predictions = (df
                          .groupBy("store", "item")
                          .apply(fit_pandas_udf)
                          )
