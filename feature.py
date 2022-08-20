import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf


# 'Age'の欠損値を平均値で補完
def fill_null_age(df):
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    return df


# 'Fare'の欠損値を平均値で補完
def fill_null_fare(df):
    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))
    return df


# 'Embarked'の欠損値をSで補完
def fill_null_embarked(df):
    df['Embarked'] = df['Embarked'].fillna('S')
    return df


# 年齢を年代に変換
def binning_age(df):
    df['Age_bin'] = pd.cut(
        df['Age'],
        bins=[0, 10, 20, 30, 40, 50, 100],
        right=False,
        labels=['U10s', '10s', '20s', '30s', '40s', 'over50s'],
        duplicates='raise',
        include_lowest=True)
    return df


# Fareを対数化
def log_fare(df):
    df['Fare_log'] = np.log1p(df['Fare'])
    return df


# 敬称
def honorific_title(df):
    honorific_title_list = ['Dona.', 'Mrs.', 'Dr.', 'Sir.', 'Major.', 'Jonkheer.', 'Col.',
                            'Countess.', 'Lady.', 'Ms.', 'Mme.', 'Miss.', 'Mlle.', 'Rev.', 'Don.', 'Master.', 'Mr.', 'Capt.']
    for n in honorific_title_list:
        df[f'{n}'] = df['Name'].apply(lambda x: 1 if f'{n}' in x else 0)
    return df


# 家族の数
def count_family(df):
    df['Family'] = df['SibSp'] + df['Parch']
    return df


# 性別ごとの旅客運賃の平均値
def mean_fare_by_sex(df):
    df['mean_Fare_by_Sex'] = df.groupby('Sex')['Fare'].transform('mean')
    return df


# 'Sex' × 'Embarked'：出現回数
def count_sex_x_embarked(df):
    df['count_Sex_x_Embarked'] = df.groupby(['Sex', 'Embarked'])[
        'PassengerId'].transform('count')
    return df

# Cabinの先頭を取り出す


def initial_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Cabin_label'] = df['Cabin'].str.get(0)
    return df


# One-Hot Encoding
def ohe(df, cfg):
    df_ohe = pd.get_dummies(df[cfg.feature.ohe_col], drop_first=False)
    df = pd.concat([df, df_ohe], axis=1)
    return df


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # データの読み込み
    train = pd.read_csv('input/titanic/train.csv')
    test = pd.read_csv('input/titanic/test.csv')

    # 学習データとテストデータの結合
    df = pd.concat([train, test], sort=False).reset_index(drop=True)

    # 前処理（欠損値補完）
    df = fill_null_age(df)
    df = fill_null_fare(df)
    df = fill_null_embarked(df)

    # 特徴量エンジニアリング
    df = binning_age(df)
    df = log_fare(df)
    df = honorific_title(df)
    df = count_family(df)
    df = mean_fare_by_sex(df)
    df = count_sex_x_embarked(df)
    df = initial_cabin(df)

    # One-Hot Encoding
    df = ohe(df, cfg)

    # trainとtestに再分割
    featured_train = df[~df['Survived'].isnull()]
    featured_test = df[df['Survived'].isnull()]

    # CSVファイルとして出力
    featured_train.to_csv(
        'input/processed/featured_train.csv', header=True, index=False)
    featured_test.to_csv(
        'input/processed/featured_test.csv', header=True, index=False)


if __name__ == '__main__':
    main()
