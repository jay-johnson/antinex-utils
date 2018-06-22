import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from spylunking.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERR
from antinex_utils.consts import NOTRUN


name = "build-scaler-dataset"
log = build_colorized_logger(name=name)


def build_scaler_dataset_from_records(
        record_list,
        label="build-scaled-dataset",
        min_feature=-1,
        max_feature=1,
        cast_to_type="float32"):
    """build_scaler_dataset_from_records
    :param record_list: list of json records to scale between min/max
    :param label: log label for tracking
    :param min_feature: min feature range for scale normalization
    :param max_feature: max feature range for scale normalization
    :param cast_to_type: cast all of the dataframe to this datatype
    """

    status = NOTRUN
    last_step = "not-run"
    df = None
    scaler = None
    dataset = None

    res = {
        "status": status,
        "err": last_step,
        "org_recs": df,
        "scaler": scaler,
        "dataset": dataset
    }

    try:
        last_step = ("building scaler range=[{},{}]").format(
            min_feature,
            max_feature)
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        scaler = MinMaxScaler(
            feature_range=(
                min_feature,
                max_feature))

        last_step = ("converting records={} to df").format(
            len(record_list))
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        df = pd.read_json(record_list)
        if cast_to_type:
            last_step = ("casting df values to type={}").format(
                cast_to_type)
            log.info(("{} - {}")
                     .format(
                        label,
                        last_step))
            only_floats = df.values.astype(cast_to_type)

        last_step = ("running scale transform rows={}").format(
            len(df.index))
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        dataset = scaler.fit_transform(
            only_floats)

        status = SUCCESS
    except Exception as e:
        last_step = ("failed build_scaler_dataset_from_records "
                     "with ex={} last_step='{}' "
                     "recs={} range=[{},{}]").format(
                        e,
                        last_step,
                        str(record_list)[0:64],
                        min_feature,
                        max_feature)

        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        status = ERR
    # end of try/ex

    res = {
        "status": status,
        "err": last_step,
        "org_recs": df,
        "scaler": scaler,
        "dataset": dataset
    }
    return res
# end of build_scaler_dataset_from_records
