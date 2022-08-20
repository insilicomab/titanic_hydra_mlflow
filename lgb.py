import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import lightgbm as lgb

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # データの読み込み
    train = pd.read_csv('input/processed/featured_train.csv')
    test = pd.read_csv('input/processed/featured_test.csv')

    # ハイパーパラメータの設定
    params = {
        'objective': cfg.lgb.params.objective,
        'random_seed': cfg.lgb.params.random_seed,
    }

    # 説明変数と目的変数を指定
    X_train = train.drop(cfg.lgb.drop_col, axis=1)
    Y_train = train['Survived']

    # Foldごとの結果を保存
    models = []
    metrics = []
    imp = pd.DataFrame()

    # K分割する
    kf = KFold(n_splits=cfg.lgb.folds)

    for nfold, (train_index, val_index) in enumerate(kf.split(X_train)):
        x_train = X_train.iloc[train_index]
        x_valid = X_train.iloc[val_index]
        y_train = Y_train.iloc[train_index]
        y_valid = Y_train.iloc[val_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(x_train,
                  y_train,
                  eval_set=(x_valid, y_valid),
                  early_stopping_rounds=cfg.lgb.model.ers,
                  verbose=cfg.lgb.model.verbose,
                  )

        y_pred = model.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        print(acc)

        models.append(model)
        metrics.append(acc)

        _imp = pd.DataFrame(
            {'col': x_train.columns, 'imp': model.feature_importances_, "nfold": nfold+1})
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    metrics = np.array(metrics)
    print(f'metrics: {np.mean(metrics):.2f} ± {np.std(metrics):.2f}')

    imp = imp.groupby("col")["imp"].agg(["mean", "std"])
    imp.columns = ["imp", "imp_std"]
    imp = imp.reset_index(drop=False).sort_values('imp', ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(imp['col'], imp['imp'], xerr=imp['imp_std'])
    plt.show()

    # 説明変数と目的変数を指定
    X_test = test.drop(cfg.lgb.drop_col, axis=1)

    # テストデータにおける予測
    preds = []

    for model in models:
        pred = model.predict(X_test)
        preds.append(pred)

    # アンサンブル学習
    preds_array = np.array(preds)
    pred = stats.mode(preds_array)[0].T  # 予測データリストのうち最頻値を算出し、行と列を入れ替え

    # 提出用ファイルの読み込み
    sub = pd.read_csv('input/titanic/gender_submission.csv')

    # 目的変数カラムの置き換え
    sub['Survived'] = pred.astype('int')

    # ファイルのエクスポート
    sub.to_csv(cfg.sub.name, index=False)


if __name__ == '__main__':
    main()
