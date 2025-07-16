from sklearn.datasets import load_breast_cancer

from ray import tune
from ray.data import from_pandas
from ray.train import RunConfig, ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from ray.tune.tuner import Tuner

def get_dataset():
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw["data"]
    dataset_df["target"] = data_raw["target"]
    dataset = from_pandas(dataset_df)
    return dataset

trainer = XGBoostTrainer(
    label_column="target",
    params={},
    datasets={"train": get_dataset()},
)

param_space = {
    "scaling_config": ScalingConfig(
        num_workers=tune.grid_search([2, 4]),
        resources_per_worker={
            "CPU": tune.grid_search([1, 2]),
        },
    ),
    # You can even grid search various datasets in Tune.
    # "datasets": {
    #     "train": tune.grid_search(
    #         [ds1, ds2]
    #     ),
    # },
    "params": {
        "objective": "binary:logistic",
        "tree_method": "approx",
        "eval_metric": ["logloss", "error"],
        "eta": tune.loguniform(1e-4, 1e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "max_depth": tune.randint(1, 9),
    },
}
tuner = Tuner(trainable=trainer, param_space=param_space,
    run_config=RunConfig(name="my_tune_run"))
results = tuner.fit()