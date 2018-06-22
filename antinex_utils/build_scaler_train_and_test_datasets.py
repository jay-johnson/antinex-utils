import pandas as pd
from spylunking.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERR
from antinex_utils.consts import FAILED
from antinex_utils.consts import NOTRUN
from antinex_utils.build_scaler_dataset_from_records import \
    build_scaler_dataset_from_records
from sklearn.model_selection import train_test_split


name = "scaler-train-test"
log = build_colorized_logger(name=name)


def build_scaler_train_and_test_datasets(
        label,
        train_features,
        test_feature,
        df,
        test_size,
        seed,
        scaler_cast_to_type="float32",
        min_feature_range=-1,
        max_feature_range=1):
    """build_scaler_train_and_test_datasets

    :param label: log label
    :param train_features: features to train
    :param test_feature: target feature name
    :param df: dataframe to build scalers and test and train datasets
    :param test_size: percent of test to train rows
    :param min_feature_range: min scaler range
    :param max_feature_range: max scaler range
    """

    status = NOTRUN
    last_step = "not-run"
    scaled_train_df = None
    scaled_test_df = None
    scaled_train_dataset = None
    scaled_test_dataset = None
    scaler_train = None
    scaler_test = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    data = {
        "scaled_train_df": scaled_train_df,
        "scaled_test_df": scaled_test_df,
        "scaled_train_dataset": scaled_train_dataset,
        "scaled_test_dataset": scaled_test_dataset,
        "scaler_train": scaler_train,
        "scaler_test": scaler_test,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "min_range": min_feature_range,
        "max_range": max_feature_range
    }
    res = {
        "status": status,
        "err": last_step,
        "data": data
    }
    try:
        last_step = ("building scalers df.rows={} columns={} "
                     "train_features={} test_feature={}").format(
                        len(df.index),
                        list(df.columns.values),
                        train_features,
                        test_feature)
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        scaled_train_df = df[train_features]
        scaled_test_df = pd.DataFrame(
            {test_feature: df[test_feature]})

        last_step = ("building scaled TRAIN dataset [{},{}]").format(
                    min_feature_range,
                    max_feature_range)
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        scaled_train_res = \
            build_scaler_dataset_from_records(
                label=label,
                record_list=scaled_train_df.to_json(),
                min_feature=min_feature_range,
                max_feature=max_feature_range,
                cast_to_type=scaler_cast_to_type)

        last_step = ("building scaled TEST dataset [{},{}]").format(
                        min_feature_range,
                        max_feature_range)
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        scaled_test_res = \
            build_scaler_dataset_from_records(
                label=label,
                record_list=scaled_test_df.to_json(),
                min_feature=min_feature_range,
                max_feature=max_feature_range,
                cast_to_type=scaler_cast_to_type)

        last_step = ("scaled dataset transform "
                     "train_status={} test_status={}").format(
                        scaled_train_res["status"] == SUCCESS,
                        scaled_test_res["status"] == SUCCESS)

        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))

        if scaled_train_res["status"] == SUCCESS \
           and scaled_test_res["status"] == SUCCESS:
            last_step = ("scaled train_rows={} "
                         "test_rows={}").format(
                            len(scaled_train_res["dataset"]),
                            len(scaled_test_res["dataset"]))
            log.info(("{} - {}")
                     .format(
                        label,
                        last_step))
            scaler_train = scaled_train_res["scaler"]
            scaler_test = scaled_test_res["scaler"]
            scaled_train_dataset = scaled_train_res["dataset"]
            scaled_test_dataset = scaled_test_res["dataset"]
            (x_train,
             x_test,
             y_train,
             y_test) = train_test_split(
                scaled_train_dataset,
                scaled_test_dataset,
                test_size=test_size,
                random_state=seed)
        else:
            last_step = ("failed dataset transform "
                         "train_status={} test_status={}").format(
                             scaled_train_res["status"],
                             scaled_test_res["status"])
            log.error(("{} - {}")
                      .format(
                        label,
                        last_step))

            status = FAILED
            res = {
                "status": status,
                "err": last_step,
                "data": data
            }
            return res
        # if built both train and test successfully

        last_step = ("train_rows={} test_rows={} "
                     "x_train={} x_test={} "
                     "y_train={} y_test={}").format(
                        len(scaled_train_df.index),
                        len(scaled_test_df),
                        len(scaled_train_df.index),
                        len(scaled_test_df.index),
                        len(scaled_train_dataset),
                        len(scaled_test_dataset))
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))

        data["scaled_train_df"] = scaled_train_df
        data["scaled_test_df"] = scaled_test_df
        data["scaled_train_dataset"] = scaled_train_dataset
        data["scaled_test_dataset"] = scaled_test_dataset
        data["scaler_train"] = scaler_train
        data["scaler_test"] = scaler_test
        data["x_train"] = x_train
        data["y_train"] = y_train
        data["x_test"] = x_test
        data["y_test"] = y_test
        status = SUCCESS
        last_step = ""

        log.info(("{} - done")
                 .format(
                    label))
    except Exception as e:
        last_step = ("failed during last_step='{}' with ex={} "
                     "building scalers df.rows={} columns={} "
                     "train_features={} test_feature={}").format(
                        last_step,
                        e,
                        len(df.index),
                        list(df.columns.values),
                        train_features,
                        test_feature)
        log.error(("{} - {}")
                  .format(
                    label,
                    last_step))
        status = ERR
    # if applying scaler to predict rows

    res = {
        "status": status,
        "err": last_step,
        "data": data
    }

    return res
# end of build_scaler_train_and_test_datasets
