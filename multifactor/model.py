# coding=utf-8
"""
多因子选股 - LightGBM排序模型

使用LightGBM进行股票收益率预测排序
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
import pickle
import os

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM未安装，使用简单线性模型替代")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class StockRanker:
    """
    股票排序模型

    使用LightGBM或线性回归预测股票未来收益，
    然后按预测值排序选股
    """

    def __init__(self, use_lgb: bool = True, model_params: dict = None):
        """
        Args:
            use_lgb: 是否使用LightGBM (False则用线性回归)
            model_params: 模型参数
        """
        self.use_lgb = use_lgb and HAS_LIGHTGBM
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        # 默认LightGBM参数
        self.lgb_params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 100,
            "early_stopping_rounds": 10
        }

        if model_params:
            self.lgb_params.update(model_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> dict:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签 (未来收益率)
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练结果统计
        """
        self.feature_names = list(X_train.columns)

        # 处理缺失值
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.use_lgb:
            # LightGBM训练
            train_data = lgb.Dataset(X_train_scaled, label=y_train)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

                self.model = lgb.train(
                    self.lgb_params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    valid_names=["train", "val"]
                )
            else:
                self.model = lgb.train(
                    self.lgb_params,
                    train_data,
                    valid_sets=[train_data],
                    valid_names=["train"]
                )

            # 特征重要性
            importance = dict(zip(self.feature_names,
                                  self.model.feature_importance(importance_type="gain")))
        else:
            # 线性回归
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_train_scaled, y_train)

            importance = dict(zip(self.feature_names,
                                  np.abs(self.model.coef_)))

        # 排序特征重要性
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

        logger.info(f"模型训练完成, Top5因子: {list(importance.keys())[:5]}")

        return {
            "feature_importance": importance,
            "n_features": len(self.feature_names),
            "n_samples": len(X_train)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测收益率

        Args:
            X: 特征DataFrame

        Returns:
            预测的收益率数组
        """
        if self.model is None:
            raise ValueError("模型未训练")

        X = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)

        if self.use_lgb:
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X_scaled)

    def rank(self, X: pd.DataFrame, codes: List[str], top_n: int = 10) -> List[str]:
        """
        排序选股

        Args:
            X: 特征DataFrame
            codes: 股票代码列表
            top_n: 选择前N只

        Returns:
            选出的股票代码列表
        """
        preds = self.predict(X)

        # 按预测值排序
        ranking = pd.DataFrame({
            "code": codes,
            "pred": preds
        }).sort_values("pred", ascending=False)

        return ranking["code"].head(top_n).tolist()

    def save(self, path: str):
        """保存模型"""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "use_lgb": self.use_lgb
            }, f)
        logger.info(f"模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self.use_lgb = data["use_lgb"]
        logger.info(f"模型已加载: {path}")


def train_ranker(factor_data: pd.DataFrame, factor_cols: List[str],
                 label_col: str = "future_ret_5d",
                 train_ratio: float = 0.8) -> Tuple[StockRanker, dict]:
    """
    训练排序模型的便捷函数

    Args:
        factor_data: 因子数据 (包含多天的截面数据)
        factor_cols: 因子列名列表
        label_col: 标签列名
        train_ratio: 训练集比例

    Returns:
        (模型, 训练结果)
    """
    # 去除缺失标签
    data = factor_data.dropna(subset=[label_col])

    if len(data) < 100:
        raise ValueError(f"样本数不足: {len(data)}")

    # 按日期划分训练集和验证集
    dates = sorted(data["date"].unique())
    split_idx = int(len(dates) * train_ratio)
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    train_data = data[data["date"].isin(train_dates)]
    val_data = data[data["date"].isin(val_dates)]

    X_train = train_data[factor_cols]
    y_train = train_data[label_col]
    X_val = val_data[factor_cols]
    y_val = val_data[label_col]

    logger.info(f"训练集: {len(train_data)} 样本, 验证集: {len(val_data)} 样本")

    # 训练模型
    ranker = StockRanker(use_lgb=HAS_LIGHTGBM)
    result = ranker.train(X_train, y_train, X_val, y_val)

    # 验证集评估
    if len(val_data) > 0:
        val_pred = ranker.predict(X_val)
        ic = np.corrcoef(val_pred, y_val)[0, 1]
        result["val_ic"] = ic
        logger.info(f"验证集IC: {ic:.4f}")

    return ranker, result
