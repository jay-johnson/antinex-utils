import json
import pandas as pd
from antinex_utils.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERR
from antinex_utils.consts import NOTRUN


name = "merge-dataset"
log = build_colorized_logger(name=name)


def merge_inverse_data_into_original(
        req,
        sort_on_index=None,
        ordered_columns=None):
    """merge_inverse_data_into_original

    :param req: managed dictionary
    :param sort_on_index: sort the dataframe on this column name
    :param ordered_columns: column list to rename the inverse transform
    """
    label = req.get(
        "label",
        "")
    last_step = "not-run"
    status = NOTRUN
    org_df = None
    predict_df = None
    res = {
        "status": status,
        "err": last_step,
        "sorted_org_df": org_df,
        "predict_df": predict_df
    }
    try:

        last_step = ("sorting org_records on index={}").format(
                        sort_on_index)
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        org_df = pd.DataFrame(req["org_recs"]).set_index(sort_on_index)
        inverse_df = pd.DataFrame(req["inverse_recs"])
        if ordered_columns:
            last_step = "getting inverse column ordering"
            log.info(("{} - {}")
                     .format(
                        label,
                        last_step))
            inverse_df.columns = ordered_columns
        if sort_on_index:
            last_step = ("sorting inverse on index={}").format(
                            sort_on_index)
            log.info(("{} - {}")
                     .format(
                        label,
                        last_step))
            inverse_df.set_index(sort_on_index)

        row_idx = 0
        predict_rows = []
        last_step = ("for org_value rows={}").format(
                        len(org_df.index))
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        for idx, row in org_df.iterrows():
            inv_row = inverse_df.iloc[row_idx].to_json()
            row_idx += 1
            predict_rows.append(
                json.loads(inv_row))
        # end of for all rows to check for ordering

        last_step = ("building predict_df rows={}").format(
                        len(predict_rows))
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        predict_df = pd.DataFrame(
            predict_rows)

        if sort_on_index:
            last_step = ("sorting={} predict_df rows={}").format(
                            sort_on_index,
                            len(predict_rows))
            log.info(("{} - {}")
                     .format(
                        label,
                        last_step))
            predict_df.set_index(sort_on_index)

        last_step = ""
        status = SUCCESS

    except Exception as e:
        last_step = ("failed merge_inverse_data_into_original "
                     "with ex={} last_step='{}' "
                     "org_recs={} inverse_recs={}").format(
                        e,
                        last_step,
                        len(org_df.index),
                        len(inverse_df.index))
        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        status = ERR
    # end of try/ex

    res = {
        "status": status,
        "err": last_step,
        "sorted_org_df": org_df,
        "predict_df": predict_df
    }

    return res
# end of merge_inverse_data_into_original
