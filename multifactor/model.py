# coding=utf-8
"""
多因子选股 - LightGBM排序模型

使用LightGBM进行股票收益率预测排序
支持GPU训练、超参数调优、交叉验证
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
import pickle
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM未安装，使用简单线性模型替代")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


def check_gpu_available() -> bool:
    """检查GPU是否可用于LightGBM"""
    if not HAS_LIGHTGBM:
        return False
    try:
        # 尝试创建GPU数据集
        test_data = lgb.Dataset(np.array([[1, 2], [3, 4]]), label=[0, 1])
        params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
        lgb.train(params, test_data, num_boost_round=1, verbose_eval=False)
        return True
    except Exception:
        return False


# 全局GPU状态
GPU_AVAILABLE = None


def is_gpu_available() -> bool:
    """懒加载检查GPU"""
    global GPU_AVAILABLE
    if GPU_AVAILABLE is None:
        GPU_AVAILABLE = check_gpu_available()
    return GPU_AVAILABLE


class StockRanker:
    """
    股票排序模型

    使用LightGBM或线性回归预测股票未来收益，
    然后按预测值排序选股
    """

    def __init__(self, use_lgb: bool = True, model_params: dict = None, use_gpu: bool = False):
        """
        Args:
            use_lgb: 是否使用LightGBM (False则用线性回归)
            model_params: 模型参数
            use_gpu: 是否使用GPU训练
        """
        self.use_lgb = use_lgb and HAS_LIGHTGBM
        self.use_gpu = use_gpu and is_gpu_available()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_history = []  # 训练历史记录

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
            "early_stopping_rounds": 10,
            "num_threads": -1,  # 使用所有CPU核心
            "force_col_wise": True  # 小数据集优化
        }

        # GPU设置
        if self.use_gpu:
            self.lgb_params.update({
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0
            })
            logger.info("启用GPU训练")

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


def cross_validate_ranker(
    factor_data: pd.DataFrame,
    factor_cols: List[str],
    label_col: str = "future_ret_5d",
    n_splits: int = 5,
    model_params: dict = None,
    use_gpu: bool = False,
    callback: Callable = None
) -> Dict[str, Any]:
    """
    时序交叉验证

    Args:
        factor_data: 因子数据
        factor_cols: 因子列名
        label_col: 标签列名
        n_splits: 折数
        model_params: 模型参数
        use_gpu: 是否用GPU
        callback: 进度回调函数 callback(fold, n_splits, ic)

    Returns:
        交叉验证结果
    """
    data = factor_data.dropna(subset=[label_col])
    dates = sorted(data["date"].unique())

    if len(dates) < n_splits * 2:
        raise ValueError(f"日期数不足: {len(dates)}, 需要至少 {n_splits * 2}")

    # 时序分割
    tscv = TimeSeriesSplit(n_splits=n_splits)
    date_indices = list(range(len(dates)))

    fold_results = []
    all_ics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(date_indices)):
        train_dates = [dates[i] for i in train_idx]
        val_dates = [dates[i] for i in val_idx]

        train_data = data[data["date"].isin(train_dates)]
        val_data = data[data["date"].isin(val_dates)]

        if len(train_data) < 50 or len(val_data) < 10:
            continue

        X_train = train_data[factor_cols]
        y_train = train_data[label_col]
        X_val = val_data[factor_cols]
        y_val = val_data[label_col]

        # 训练
        ranker = StockRanker(use_lgb=HAS_LIGHTGBM, model_params=model_params, use_gpu=use_gpu)
        result = ranker.train(X_train, y_train, X_val, y_val)

        # 验证IC
        val_pred = ranker.predict(X_val)
        ic = np.corrcoef(val_pred, y_val)[0, 1]
        all_ics.append(ic)

        fold_results.append({
            "fold": fold + 1,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "ic": ic,
            "feature_importance": result.get("feature_importance", {})
        })

        if callback:
            callback(fold + 1, n_splits, ic)

        logger.info(f"Fold {fold + 1}/{n_splits}: IC = {ic:.4f}")

    # 汇总
    mean_ic = np.mean(all_ics) if all_ics else 0
    std_ic = np.std(all_ics) if all_ics else 0

    # 合并特征重要性
    merged_importance = {}
    for r in fold_results:
        for feat, imp in r.get("feature_importance", {}).items():
            if feat not in merged_importance:
                merged_importance[feat] = []
            merged_importance[feat].append(imp)

    avg_importance = {k: np.mean(v) for k, v in merged_importance.items()}
    avg_importance = dict(sorted(avg_importance.items(), key=lambda x: -x[1]))

    return {
        "n_splits": n_splits,
        "fold_results": fold_results,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_list": all_ics,
        "avg_feature_importance": avg_importance
    }


def optimize_hyperparams(
    factor_data: pd.DataFrame,
    factor_cols: List[str],
    label_col: str = "future_ret_5d",
    n_trials: int = 50,
    n_cv_splits: int = 3,
    use_gpu: bool = False,
    callback: Callable = None
) -> Dict[str, Any]:
    """
    使用Optuna进行超参数调优

    Args:
        factor_data: 因子数据
        factor_cols: 因子列名
        label_col: 标签列名
        n_trials: 试验次数
        n_cv_splits: 交叉验证折数
        use_gpu: 是否用GPU
        callback: 进度回调 callback(trial_num, n_trials, best_ic)

    Returns:
        最优参数和结果
    """
    if not HAS_OPTUNA:
        raise ImportError("需要安装optuna: pip install optuna")

    data = factor_data.dropna(subset=[label_col])
    dates = sorted(data["date"].unique())

    # 定义搜索空间
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        # 交叉验证
        try:
            cv_result = cross_validate_ranker(
                factor_data=data,
                factor_cols=factor_cols,
                label_col=label_col,
                n_splits=n_cv_splits,
                model_params=params,
                use_gpu=use_gpu
            )
            mean_ic = cv_result["mean_ic"]
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            mean_ic = -1.0

        if callback:
            callback(trial.number + 1, n_trials, mean_ic)

        return mean_ic

    # 运行优化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_ic = study.best_value

    logger.info(f"最优参数: {best_params}")
    logger.info(f"最优IC: {best_ic:.4f}")

    return {
        "best_params": best_params,
        "best_ic": best_ic,
        "n_trials": n_trials,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
    }


def train_and_save_model(
    factor_data: pd.DataFrame,
    factor_cols: List[str],
    save_dir: str,
    model_name: str = None,
    label_col: str = "future_ret_5d",
    model_params: dict = None,
    use_gpu: bool = False,
    train_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    训练并保存模型

    Args:
        factor_data: 因子数据
        factor_cols: 因子列名
        save_dir: 保存目录
        model_name: 模型名称（不提供则自动生成）
        label_col: 标签列名
        model_params: 模型参数
        use_gpu: 是否用GPU
        train_ratio: 训练集比例

    Returns:
        训练结果和保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 生成模型名称
    if model_name is None:
        model_name = f"lgb_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 训练
    data = factor_data.dropna(subset=[label_col])
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

    ranker = StockRanker(use_lgb=HAS_LIGHTGBM, model_params=model_params, use_gpu=use_gpu)
    result = ranker.train(X_train, y_train, X_val, y_val)

    # 验证IC
    val_pred = ranker.predict(X_val)
    ic = np.corrcoef(val_pred, y_val)[0, 1]
    result["val_ic"] = ic

    # 保存模型
    model_path = os.path.join(save_dir, f"{model_name}.pkl")
    ranker.save(model_path)

    # 保存元数据
    meta = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "factor_cols": factor_cols,
        "label_col": label_col,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "val_ic": float(ic),
        "model_params": model_params or {},
        "use_gpu": use_gpu,
        "feature_importance": {k: float(v) for k, v in result.get("feature_importance", {}).items()}
    }

    meta_path = os.path.join(save_dir, f"{model_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存: {model_path}")

    return {
        "model_path": model_path,
        "meta_path": meta_path,
        "result": result,
        "val_ic": ic,
        "meta": meta
    }


def load_saved_model(model_path: str) -> Tuple[StockRanker, Dict]:
    """
    加载保存的模型

    Args:
        model_path: 模型文件路径

    Returns:
        (模型, 元数据)
    """
    ranker = StockRanker()
    ranker.load(model_path)

    # 尝试加载元数据
    meta_path = model_path.replace(".pkl", "_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return ranker, meta


def list_saved_models(save_dir: str) -> List[Dict]:
    """
    列出保存的模型

    Args:
        save_dir: 模型目录

    Returns:
        模型列表
    """
    if not os.path.exists(save_dir):
        return []

    models = []
    for f in os.listdir(save_dir):
        if f.endswith("_meta.json"):
            meta_path = os.path.join(save_dir, f)
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
                meta["meta_path"] = meta_path
                meta["model_path"] = meta_path.replace("_meta.json", ".pkl")
                models.append(meta)

    # 按创建时间排序
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return models
