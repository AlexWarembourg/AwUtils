def get_features_pipeline(training_features, categorical_col):
    """
    Preprocessing pipeline before GBDT
    :param training_features: a list of columns for training
    :param categorical_col: a list of columns to be indexed
    :return: A pipeline object
    """
    indexers = [StringIndexer(
        inputCol=c, outputCol="{}_indexedd".format(c), handleInvalid='skip'
    ) for c in categorical_col]
    all_features = training_features + [ind.getOutputCol() for ind in indexers]
    assembler = VectorAssembler(inputCols=all_features, outputCol="features")
    return Pipeline(stages=indexers + [assembler])


def save_features_importances(model, training_features):
    """
    Get features importances from model inside Pipeline
    :param model: model object
    :return: a sorted dictionnary with features as key and importances as values
    """
    fi = model.stages[0].featureImportances.toArray()
    feat_imp = dict(zip(training_features, fi))
    sorted_x = sorted(feat_imp.items(), key=lambda kv: kv[1])
    return sc.parallelize(sorted_x).coalesce(1).saveAsTextFile("gs://path/feat_imp/*.txt")


@udf(FloatType())
def single_autocorr(s1, s2):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    import numpy as np
    if s2 is not None:
        ms1 = np.mean(s1)
        ms2 = np.mean(s2)
        ds1 = s1 - ms1
        ds2 = s2 - ms2
        divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
        return float(np.sum(ds1 * ds2) / divider) if divider != 0.0 else 0.0
    else:
        -2.0


# TODO : Make it more generic ...
def get_promotional_autocorr(df_train):
    keys = ["prd_sku_unique_code", "cial_offer_code", "previous_cial_offer_code",
            "site_unique_code", "flag_loyalty", "nb_prod", "dt_ticket_sale"]
    # select col
    X = df_train.select(*keys).drop_duplicates().cache()
    # define grouped cols
    needed_col = ['prd_sku_unique_code', 'flag_loyalty', 'site_unique_code', 'cial_offer_code']
    # define windows to agg
    w = Window.partitionBy(*needed_col).orderBy('dt_ticket_sale')
    # get array of qty
    last_ope_sorted = (X
                       .withColumn('last_operation', F.collect_list('nb_prod').over(w))
                       .groupBy(*needed_col)
                       .agg(F.max('last_operation').alias('last_operation'))
                       )
    # split for join
    curr = (last_ope_sorted.select(
        col("cial_offer_code").alias("current_cial_offer_code"),
        col('last_operation').alias('current_operation'),
        col('prd_sku_unique_code').alias('current_prd_sku_unique_code'),
        col('site_unique_code').alias('current_site_unique_code'),
        col('flag_loyalty').alias('current_flag_loyalty')))
    # split for join
    previous = (last_ope_sorted.select(
        col("cial_offer_code").alias("prev_cial_offer_code"),
        col('last_operation').alias('previous_operation'),
        col('prd_sku_unique_code').alias('previous_prd_sku_unique_code'),
        col('site_unique_code').alias('previous_site_unique_code'),
        col('flag_loyalty').alias('previous_flag_loyalty')))
    # make join of both prev and curr offer to have [curr array] , [prev array]
    operations_df = (X
                     .join(curr,
                           [(curr.current_cial_offer_code == X.cial_offer_code) &
                            (curr.current_flag_loyalty == X.flag_loyalty) &
                            (curr.current_prd_sku_unique_code == X.prd_sku_unique_code) &
                            (curr.current_site_unique_code == X.site_unique_code)],
                           how="left"
                           )
                     .join(previous,
                           [(previous.prev_cial_offer_code == X.previous_cial_offer_code) &
                            (previous.previous_flag_loyalty == X.flag_loyalty) &
                            (previous.previous_prd_sku_unique_code == X.prd_sku_unique_code) &
                            (previous.previous_site_unique_code == X.site_unique_code)],
                           how="left"
                           )
                     .select("prd_sku_unique_code", "cial_offer_code", "previous_cial_offer_code", "site_unique_code",
                             "flag_loyalty", "dt_ticket_sale", "previous_operation", "current_operation"
                             )
                     .groupBy("prd_sku_unique_code", "cial_offer_code", "previous_cial_offer_code", "site_unique_code",
                              "flag_loyalty")
                     .agg(F.max('current_operation').alias('current_operation'),
                          F.max('previous_operation').alias('previous_operation'))
                     .dropna()
                     .withColumn('curr_shape', F.size('current_operation'))
                     .withColumn('prev_shape', F.size('previous_operation'))
                     .withColumn('matching_shape', F.when(col('curr_shape') == col('prev_shape'), 1).otherwise(0))
                     .filter(col('matching_shape') == 1)
                     .withColumn('series_autocorr', F.when(col("previous_operation").isNotNull(),
                                                           single_autocorr(
                                                               F.array(col('current_operation')),
                                                               F.array(col('previous_operation'))
                                                           )).otherwise(
        F.lit(-2.0))
                                 )
                     )
    # return series autocorr
    return_key = ["prd_sku_unique_code", "site_unique_code", "cial_offer_code", "flag_loyalty", "series_autocorr"]
    operations_df = (operations_df
                     .select(*return_key)
                     .groupBy('prd_sku_unique_code', 'site_unique_code')
                     .agg(avg(col("series_autocorr")).alias('series_autocorr'))
                     )

    return operations_df