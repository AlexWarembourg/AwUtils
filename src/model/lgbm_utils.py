import lightgbm  as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def lgb_dataset(train, test, target, features):
    lgb_train = lgb.Dataset(train[features].values, train[target].values)
    lgb_eval = lgb.Dataset(test[features].values, test[target].values, reference=lgb_train)
    return lgb_train, lgb_eval


def plot_lgbm(gbm, features_used, png_save=True):
    feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(),
                                          features_used)),
                               columns=['Value', 'Feature'])
    if png_save:
        with plt.xkcd():
            plt.figure(figsize=(18, 9))
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
            plt.title('LightGBM Features (avg over folds)')
            plt.tight_layout()
            plt.savefig("gbm_plot.png", format="png")

    else:
        with plt.xkcd():
            plt.figure(figsize=(18, 9))
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
            plt.title('LightGBM Features (avg over folds)')
            plt.tight_layout()
            plt.show()
