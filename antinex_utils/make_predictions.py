import os
import uuid
import json
import numpy
import pandas as pd
import copy
from spylunking.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERR
from antinex_utils.consts import FAILED
from antinex_utils.consts import NOTRUN
from antinex_utils.utils import ev
from antinex_utils.utils import ppj
from antinex_utils.build_training_request import \
    build_training_request
from antinex_utils.build_scaler_dataset_from_records import \
    build_scaler_dataset_from_records
from antinex_utils.build_scaler_train_and_test_datasets import \
    build_scaler_train_and_test_datasets
from antinex_utils.merge_inverse_data_into_original import \
    merge_inverse_data_into_original
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt  # noqa


name = "make-predict"
log = build_colorized_logger(name=name)


def build_regression_dnn(
        num_features,
        compile_data,
        label="",
        model_json=None,
        model_desc=None):
    """build_regression_dnn

    :param num_features: input_dim for the number of
                         features in the data
    :param compile_data: dictionary of compile options
    :param label: log label for tracking this method
    :param model_json: keras model json to build the model
    :param model_desc: optional dictionary for model
    """

    model = Sequential()

    if model_json:
        log.info(("{} building regression "
                  "dnn model_json={}")
                 .format(
                     label,
                     model_json))
        model = model_from_json(json.dumps(model_json))
    elif model_desc:
        log.info(("{} building regression "
                  "dnn num_features={} model_desc={}")
                 .format(
                     label,
                     num_features,
                     model_desc))
        num_layers = 0
        for idx, node in enumerate(model_desc["layers"]):
            layer_type = node.get(
                "layer_type",
                "dense").lower()
            if layer_type == "dense":
                if num_layers == 0:
                    model.add(
                        Dense(
                            int(node["num_neurons"]),
                            input_dim=num_features,
                            kernel_initializer=node["init"],
                            activation=node["activation"]))
                else:
                    model.add(
                        Dense(
                            int(node["num_neurons"]),
                            kernel_initializer=node["init"],
                            activation=node["activation"]))
            else:
                if layer_type == "dropout":
                    model.add(
                        Dropout(
                            float(node["rate"])))
            # end of supported model types
            num_layers += 1
        # end of all layers
    else:
        # https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/  # noqa
        log.info(("{} building regression "
                  "dnn num_features={}")
                 .format(
                     label,
                     num_features))
        model.add(
            Dense(
                8,
                input_dim=num_features,
                kernel_initializer="normal",
                activation="relu"))
        model.add(
            Dense(
                6,
                kernel_initializer="normal",
                activation="relu"))
        model.add(
            Dense(
                1,
                kernel_initializer="normal"))
    # end of building a regression dnn

    # if model was defined
    if model:
        log.info(("{} - regression compiling={}")
                 .format(
                    label,
                    compile_data))
        # compile the model
        loss = compile_data.get(
            "loss",
            "mse")
        optimizer = compile_data.get(
            "optimizer",
            "adam")
        metrics = compile_data.get(
            "metrics",
            [
                "mse",
                "mae",
                "mape",
                "cosine"
            ])
        model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    else:
        log.error(("{} - failed building regression model")
                  .format(
                      label))
    # if could compile model

    return model
# end of build_regression_dnn


def build_classification_dnn(
        num_features,
        compile_data,
        label="",
        model_json=None,
        model_desc=None):
    """build_classification_dnn

    :param num_features: input_dim for the number of
                         features in the data
    :param compile_data: dictionary of compile options
    :param label: log label for tracking this method
    :param model_json: keras model json to build the model
    :param model_desc: optional dictionary for model
    """

    model = Sequential()

    if model_json:
        log.info(("{} building classification "
                  "dnn model_json={}")
                 .format(
                     label,
                     num_features,
                     model_json))
        model = model_from_json(json.dumps(model_json))
    elif model_desc:
        log.info(("{} building classification "
                  "dnn num_features={} model_desc={}")
                 .format(
                     label,
                     num_features,
                     model_desc))
        num_layers = 0
        for idx, node in enumerate(model_desc["layers"]):
            layer_type = node.get(
                "layer_type",
                "dense").lower()
            if layer_type == "dense":
                if num_layers == 0:
                    model.add(
                        Dense(
                            int(node["num_neurons"]),
                            input_dim=num_features,
                            kernel_initializer=node["init"],
                            activation=node["activation"]))
                else:
                    model.add(
                        Dense(
                            int(node["num_neurons"]),
                            kernel_initializer=node["init"],
                            activation=node["activation"]))
            else:
                if layer_type == "dropout":
                    model.add(
                        Dropout(
                            float(node["rate"])))
            # end of supported model types
            num_layers += 1
        # end of all layers
    else:
        # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/  # noqa
        log.info(("{} building classification "
                  "dnn num_features={}")
                 .format(
                     label,
                     num_features))
        model.add(
            Dense(
                8,
                input_dim=num_features,
                kernel_initializer="uniform",
                activation="relu"))
        model.add(
            Dense(
                6,
                kernel_initializer="uniform",
                activation="relu"))
        model.add(
            Dense(
                1,
                kernel_initializer="uniform",
                activation="sigmoid"))
    # end of building a classification dnn

    # if model was defined
    if model:
        log.info(("{} - classification compiling={}")
                 .format(
                    label,
                    compile_data))
        # compile the model
        loss = compile_data.get(
            "loss",
            "binary_crossentropy")
        optimizer = compile_data.get(
            "optimizer",
            "adam")
        metrics = compile_data.get(
            "metrics",
            [
                "accuracy"
            ])
        model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    else:
        log.error(("{} - failed building classification model")
                  .format(
                      label))
    # if could compile model

    return model
# end of build_classification_dnn


def check_request(
        req):
    """check_request

    :param req: dictionary to check values
    """
    label = req.get("label", "no-label-set")
    predict_rows = req.get("predict_rows", None)
    manifest = req.get("manifest", None)
    dataset = req.get("dataset", None)
    csv_file = req.get("csv_file", None)
    use_existing_model = req.get("use_existing_model", False)

    if use_existing_model:
        if not predict_rows and not dataset:
            return ("{} missing predict_rows or dataset for existing model"
                    "request={}").format(
                        label,
                        req)
        else:
            return None
    # existing models just need a couple rows or a dataset to work

    if not manifest and not dataset and not predict_rows and not csv_file:
        return ("{} missing manifest "
                "request={}").format(
                    label,
                    ppj(req))
    if manifest:
        csv_file = manifest.get(
            "csv_file",
            None)
        if not predict_rows and not csv_file and not dataset:
            return ("{} missing dataset predict_rows or csv_file in "
                    "manifest of request={}").format(
                        label,
                        ppj(req))

    if not dataset and not predict_rows and not csv_file:
        return ("{} missing dataset or predict_rows or csv_file in "
                "request={}").format(
                    label,
                    ppj(req))

    return None
# end of check_request


def save_prediction_image(
        label="not-set",
        history=None,
        histories=[],
        image_file=None):
    """save_prediction_image

    :param history: model prediction history
    :param histories: histories to generate in the image
    :param image_file: save to file
    """

    status = FAILED

    if not history:
        log.info(("{} - no history")
                 .format(
                     label))
        return status
    if not histories:
        log.info(("{} - no histories")
                 .format(
                     label))
        return status
    if not image_file:
        log.info(("{} - no image_file")
                 .format(
                     label))
        return status

    try:
        if history and len(histories) > 0:
            log.info(("plotting history={} "
                      "histories={}")
                     .format(
                        history,
                        histories))
            should_save = False
            for h in histories:
                if h in history.history:
                    log.info(("plotting={}")
                             .format(
                                h))
                    plt.plot(
                        history.history[h],
                        label=h)
                    should_save = True
                else:
                    log.error(("missing history={}")
                              .format(
                                h))
            # for all histories

            if should_save:
                log.info(("saving plots as image={}")
                         .format(
                            image_file))
                plt.legend(loc='best')
                plt.savefig(image_file)
                if not os.path.exists(image_file):
                    log.error(("Failed saving image={}")
                              .format(
                                    image_file))
            # end of saving file

        # end of if there are histories to plot
    except Exception as e:
        log.error(("Failed saving "
                   "image_file={} ex={}")
                  .format(
                    image_file,
                    e))
    # end of try/ex
# end of save_prediction_image


def build_train_and_test_features(
        df_columns,
        features_to_process,
        predict_feature,
        ignore_features):
    """build_train_and_test_features

    Order matters when slicing up datasets using scalers...

    if not, then something that is an int/bool can get into a float
    column and that is really bad for making predictions
    with new or pre-trained models...

    :param df_columns: columns in the dataframe
    :param features_to_process: requested features to train
    :param predict_feature: requested feature to predict
    :param ignore_features: requested non-numeric/not-wanted features
    """
    train_and_test_features = []

    # for all columns in the data
    # add columns in order if they are in the requested:
    # features_to_process list and not in the ignore_features
    for c in df_columns:
        if c == predict_feature:
            train_and_test_features.append(
                c)
        else:
            add_feature = True
            for i in ignore_features:
                if i == c:
                    add_feature = False
                    break
            if add_feature:
                for f in features_to_process:
                    if f == c:
                        train_and_test_features.append(
                            c)
                        break
    # end of filtering features before scalers

    return train_and_test_features
# end of build_train_and_test_features


def make_predictions(
        req):
    """make_predictions

    :param req: dictionary for making predictions
    """

    last_step = "not-run"
    label = "no-label-set"
    model = None
    predictions = None
    sample_predictions = []
    rounded = []
    accuracy = None
    error = None
    image_file = None
    history = None
    histories = None
    indexes = None
    scores = None
    cm = None
    scaler_res = None
    scaler_res_data = None
    scaler_train = None
    scaler_test = None
    scaled_train_dataset = None
    scaled_test_dataset = None
    inverse_predictions = None
    merge_df = None
    are_predicts_merged = False
    existing_model_dict = None
    data = {
        "predictions": predictions,
        "rounded_predictions": rounded,
        "sample_predictions": sample_predictions,
        "acc": accuracy,
        "scores": scores,
        "history": history,
        "histories": histories,
        "image_file": image_file,
        "model": model,
        "indexes": indexes,
        "confusion_matrix": cm,
        "scaler_train": scaler_train,
        "scaler_test": scaler_test,
        "scaled_train_dataset": scaled_train_dataset,
        "scaled_test_dataset": scaled_test_dataset,
        "inverse_predictions": inverse_predictions,
        "apply_scaler": False,
        "are_predicts_merged": are_predicts_merged,
        "merge_df": merge_df,
        "err": error
    }
    res = {
        "status": NOTRUN,
        "err": last_step,
        "data": None
    }

    try:

        label = req.get("label", "no-label-set")
        last_step = "validating"
        log.info("{} - {}".format(
            label,
            last_step))

        invalid_error_string = check_request(req)
        if invalid_error_string:
            last_step = ("{} - {}").format(
                            label,
                            invalid_error_string)
            log.info(("predictions stopping: {}")
                     .format(
                        last_step))
            res["err"] = last_step
            res["status"] = ERR
            res["data"] = None
            return res
        # end of checking for bad request inputs

        predict_rows = req.get("predict_rows", None)
        verbose = int(req.get("verbose", "1"))
        manifest = req.get("manifest", {})
        model_json = req.get("model_json", None)
        model_desc = req.get("model_desc", None)
        weights_json = req.get("weights_json", None)
        weights_file = req.get("weights_file", None)
        should_predict = req.get("should_predict", True)
        dataset = req.get("dataset", None)
        new_model = True
        existing_model_dict = req.get("use_existing_model", None)
        if existing_model_dict:
            log.info(("{} - using existing model={}")
                     .format(
                        label,
                        existing_model_dict["model"]))
            accuracy = existing_model_dict["acc"]
            scores = existing_model_dict["scores"]
            history = existing_model_dict["history"]
            histories = existing_model_dict["histories"]
            model = existing_model_dict["model"]
            new_model = False
        else:
            log.info(("{} - new model")
                     .format(
                        label))
        # end of loading the existing model

        save_weights = False
        image_file = req.get("image_file", None)
        loss = req.get(
            "loss",
            manifest.get(
                "loss",
                "mse"))
        optimizer = req.get(
            "optimizer",
            manifest.get(
                "optimizer",
                "adam"))
        metrics = req.get(
            "metrics",
            manifest.get(
                "metrics",
                [
                    "accuracy"
                ]))
        histories = req.get(
            "histories",
            manifest.get(
                "histories",
                [
                    "val_loss",
                    "val_acc",
                    "loss",
                    "acc"
                ]))
        ml_type = req.get(
            "ml_type",
            manifest.get(
                "ml_type",
                "classification"))
        features_to_process = req.get(
            "features_to_process",
            manifest.get(
                "features_to_process",
                []))
        filter_features = req.get(
            "filter_features",
            [])
        ignore_features = req.get(
            "ignore_features",
            [])
        predict_feature = req.get(
            "predict_feature",
            manifest.get(
                "predict_feature",
                None))
        epochs = int(req.get(
            "epochs",
            manifest.get(
                "epochs",
                "5")))
        batch_size = int(req.get(
            "batch_size",
            manifest.get(
                "batch_size",
                "32")))
        test_size = float(req.get(
            "test_size",
            manifest.get(
                "test_size",
                "0.2")))
        num_splits = int(req.get(
            "num_splits",
            manifest.get(
                "num_splits",
                "2")))
        verbose = int(req.get(
            "verbose",
            manifest.get(
                "verbose",
                "1")))
        seed = int(req.get(
            "seed",
            manifest.get(
                "seed",
                "9")))
        label_rules = req.get(
            "label_rules",
            manifest.get(
                "label_rules",
                {}))
        min_scaler_range = int(req.get(
            "min_scaler_range",
            "-1"))
        max_scaler_range = int(req.get(
            "max_scaler_range",
            "1"))
        apply_scaler = bool(str(req.get(
            "apply_scaler",
            "false")).lower() == "true")
        scaler_cast_to_type = req.get(
            "scaler_cast_type",
            "float32")
        sort_by = req.get(
            "sort_values",
            None)
        max_records = int(req.get(
            "max_records",
            "100000"))
        predict_type = manifest.get(
            "predict_type",
            "predict")
        manifest_headers = manifest.get(
            "headers",
            [])
        csv_file = manifest.get(
            "csv_file",
            None)
        meta_file = manifest.get(
            "meta_file",
            None)
        if not weights_file:
            weights_file = manifest.get(
                "model_weights_file",
                None)
        num_features = len(features_to_process)
        detected_headers = []
        num_samples = None
        org_df = None
        row_df = None
        filter_df = None
        sample_rows = None
        target_rows = None
        num_target_rows = None
        ml_req = None
        use_evaluate = False
        if csv_file and meta_file and predict_feature:
            if os.path.exists(csv_file) and os.path.exists(meta_file):
                use_evaluate = True
        else:
            if dataset:
                if os.path.exists(dataset):
                    use_evaluate = True
                    csv_file = dataset
            else:
                if predict_rows and not existing_model_dict:
                    use_evaluate = True
        # end of if we're building a dataset from these locations

        numpy.random.seed(seed)

        last_step = ("loading prediction "
                     "into dataframe seed={} "
                     "scaler={} range[{},{}]").format(
                        seed,
                        apply_scaler,
                        min_scaler_range,
                        max_scaler_range)
        log.info("{} - {}".format(
            label,
            last_step))
        if not weights_file:
            weights_file = manifest.get(
                "model_weights_file",
                None)

        last_step = ("loading prediction into dataframe")

        log.info("{} - {}".format(
            label,
            last_step))

        # convert json into pandas dataframe for model.predict
        try:
            if new_model and use_evaluate and not predict_rows:
                log.info(("{} - loading predictions new_model={} "
                          "evaluate={} csv={} sort={}")
                         .format(
                            label,
                            new_model,
                            use_evaluate,
                            csv_file,
                            sort_by))
                if sort_by:
                    org_df = pd.read_csv(
                            csv_file,
                            encoding="utf-8-sig").sort_values(
                            by=sort_by)
                else:
                    org_df = pd.read_csv(
                            csv_file,
                            encoding="utf-8-sig")

                predict_rows = org_df.to_json()
                detected_headers = list(org_df.columns.values)

                if apply_scaler:

                    for f in features_to_process:
                        if f not in org_df.columns:
                            log.error(("{} "
                                       "csv={} is missing column={}")
                                      .format(
                                        label,
                                        csv_file,
                                        f))
                    # show columns that were supposed to be in the
                    # dataset but are not

                    train_and_test_features = \
                        build_train_and_test_features(
                            org_df.columns,
                            features_to_process,
                            predict_feature,
                            ignore_features)

                    log.info(("building csv scalers all_features={}")
                             .format(
                                train_and_test_features))

                    scaler_transform_res = \
                        build_scaler_dataset_from_records(
                            label=label,
                            record_list=org_df[
                                train_and_test_features].to_json(),
                            min_feature=min_scaler_range,
                            max_feature=max_scaler_range,
                            cast_to_type=scaler_cast_to_type)

                    if scaler_transform_res["status"] == SUCCESS:
                        log.info(("{} - scaled dataset predict_rows={} "
                                  "df={} dataset={}")
                                 .format(
                            label,
                            len(predict_rows),
                            len(scaler_transform_res["org_recs"].index),
                            len(scaler_transform_res["dataset"])))
                    else:
                        log.error(("{} - failed to scale dataset err={} "
                                   "predict_rows={} df={} dataset={}")
                                  .format(
                            label,
                            scaler_transform_res["err"],
                            len(predict_rows),
                            len(scaler_transform_res["org_recs"].index),
                            len(scaler_transform_res["dataset"])))
                    # if scaler works on dataset

                    log.info(("{} building scaled samples and rows")
                             .format(
                                label))
                    # noqa https://stackoverflow.com/questions/21764475/scaling-numbers-column-by-column-with-pandas-python
                    row_df = pd.DataFrame(
                                scaler_transform_res["dataset"],
                                columns=org_df[
                                    train_and_test_features].columns)
                    sample_rows = row_df[features_to_process]
                    target_rows = row_df[predict_feature]
                    num_samples = len(sample_rows.index)
                    num_target_rows = len(target_rows.index)
                else:
                    log.info(("{} - not applying scaler to predict_rows")
                             .format(
                                label))
                    row_df = org_df
                    log.info(("{} - setting samples "
                              "to features_to_process={} cols={}")
                             .format(
                                label,
                                ppj(features_to_process),
                                list(org_df.columns.values)))
                    sample_rows = row_df[features_to_process]
                    target_rows = row_df[predict_feature]
                    num_samples = len(sample_rows.index)
                    num_target_rows = len(target_rows.index)
                # if applying scaler to predict rows
            # end of loading from a csv

            if dataset:
                log.info(("{} loading dataset={}")
                         .format(
                            label,
                            dataset))

                if sort_by:
                    org_df = pd.read_csv(
                        dataset,
                        encoding="utf-8-sig").sort_values(
                            by=sort_by)
                else:
                    org_df = pd.read_csv(
                        dataset,
                        encoding="utf-8-sig")
                # end of loading the df

                log.info(("{} preparing dataset={}")
                         .format(
                            label,
                            dataset))
                ml_req = {
                    "X_train": None,
                    "Y_train": None,
                    "X_test": None,
                    "Y_test": None
                }
                scaled_train_features = []
                cur_headers = list(org_df.columns.values)
                for h in cur_headers:
                    include_feature = True
                    if h == predict_feature:
                        include_feature = False
                    else:
                        for f in features_to_process:
                            if h == f:
                                scaled_train_features.append(h)
                                include_feature = False
                                break
                        for e in ignore_features:
                            if h == e:
                                include_feature = False
                                break
                    # filter out columns

                    if include_feature:
                        features_to_process.append(h)
                # end of building features

                # make sure to prune out ignored ones:
                cleaned_scaled_train = []
                for idx, h in enumerate(scaled_train_features):
                    should_include = True
                    for i in ignore_features:
                        if h == i:
                            should_include = False
                    if should_include:
                        cleaned_scaled_train.append(h)
                # end of pruning ignored ones

                # assign to cleaned list
                scaled_train_features = cleaned_scaled_train

                filter_features = copy.deepcopy(features_to_process)
                include_predict_feature = True
                for f in filter_features:
                    if f == predict_feature:
                        include_predict_feature = False
                        break

                if include_predict_feature:
                    filter_features.append(predict_feature)

                if apply_scaler:

                    num_features = len(scaled_train_features)
                    log.info(("{} scaling dataset={} "
                              "scaled_train_features={}")
                             .format(
                                label,
                                len(org_df.index),
                                ppj(scaled_train_features)))

                    scaler_res = \
                        build_scaler_train_and_test_datasets(
                            label=label,
                            train_features=scaled_train_features,
                            test_feature=predict_feature,
                            df=org_df,
                            test_size=test_size,
                            seed=seed,
                            scaler_cast_to_type=scaler_cast_to_type,
                            min_feature_range=min_scaler_range,
                            max_feature_range=max_scaler_range)

                    if scaler_res["status"] != SUCCESS:
                        log.info(("{} - scaler transform failed error={}")
                                 .format(
                                    label,
                                    scaler_res["err"]))
                        res["status"] = ERR
                        res["err"] = last_step
                        res["data"] = None
                        return res
                    else:
                        log.info(("{} - scaler transform done")
                                 .format(
                                    label))
                        scaler_res_data = scaler_res["data"]
                        ml_req["X_train"] = scaler_res_data["x_train"]
                        ml_req["Y_train"] = scaler_res_data["y_train"]
                        ml_req["X_test"] = scaler_res_data["x_test"]
                        ml_req["Y_test"] = scaler_res_data["y_test"]

                        scaler_train = scaler_res_data["scaler_train"]
                        scaler_test = scaler_res_data["scaler_test"]
                        scaled_train_dataset = \
                            scaler_res_data["scaled_train_dataset"]
                        scaled_test_dataset = \
                            scaler_res_data["scaled_test_dataset"]

                        log.info(("{} - building scaled row_df "
                                  "filter_features={}")
                                 .format(
                                    label,
                                    filter_features))

                        # noqa https://stackoverflow.com/questions/21764475/scaling-numbers-column-by-column-with-pandas-python
                        last_step = ("building row_df from scaled ds "
                                     "train_features={}").format(
                                        scaled_train_features)
                        row_df = pd.DataFrame(
                            scaler_res_data["scaled_train_dataset"],
                            columns=list(scaled_train_features))
                        last_step = ("building samples from rows_df={} "
                                     "train_features={}").format(
                                        len(row_df.index),
                                        scaled_train_features)
                        sample_rows = pd.DataFrame(
                            scaler_res_data["scaled_train_dataset"],
                            columns=list(scaled_train_features))
                        last_step = ("building targets from scaled ds "
                                     "predict_feature={}").format(
                                        predict_feature)
                        target_rows = pd.DataFrame(
                            scaler_res_data["scaled_test_dataset"],
                            columns=[predict_feature])
                        last_step = ("adding predict_feature={} to "
                                     "row_df").format(
                                        predict_feature)
                        row_df[predict_feature] = \
                            scaler_res_data["scaled_test_dataset"]
                        last_step = ("counting num_sample_rows")
                        num_samples = len(sample_rows.index)
                        last_step = ("counting num_target_rows")
                        num_target_rows = len(target_rows.index)
                        log.info(("{} row_df created features={} "
                                  "num_samples={} num_targets={}")
                                 .format(
                                    label,
                                    features_to_process,
                                    num_samples,
                                    num_target_rows))
                    # end of setting up scaler train/test data
                else:
                    num_features = len(features_to_process)
                    log.info(("{} filtering dataset={} filter_features={}")
                             .format(
                                label,
                                len(org_df.index),
                                ppj(filter_features)))
                    filter_df = org_df[filter_features]

                    log.info(("{} splitting non-scaled"
                              "filtered_df={} predict_feature={} test_size={} "
                              "features={} ignore_features={} csv={}")
                             .format(
                                label,
                                len(filter_df.index),
                                test_size,
                                predict_feature,
                                ppj(features_to_process),
                                ppj(ignore_features),
                                dataset))
                    # split the data into training

                    (ml_req["X_train"],
                     ml_req["X_test"],
                     ml_req["Y_train"],
                     ml_req["Y_test"]) = train_test_split(
                        filter_df[features_to_process],
                        filter_df[predict_feature],
                        test_size=test_size,
                        random_state=seed)

                    row_df = org_df
                    log.info(("{} - setting samples "
                              "to features_to_process={} cols={}")
                             .format(
                                label,
                                ppj(features_to_process),
                                list(org_df.columns.values)))
                    sample_rows = row_df[features_to_process]
                    target_rows = row_df[predict_feature]
                    num_samples = len(sample_rows.index)
                    num_target_rows = len(target_rows.index)
                # if applying scaler to predict rows
            else:
                if apply_scaler:
                    log.info(("{} - no dataset - scaling predict_rows={}")
                             .format(
                                label,
                                len(predict_rows)))

                    org_df = pd.read_json(predict_rows)
                    row_df = org_df

                    for f in features_to_process:
                        if f not in org_df.columns:
                            log.error(("{} "
                                       "predict_rows are missing column={}")
                                      .format(
                                        label,
                                        f))
                    # show columns that were supposed to be in the
                    # predict_rows but are not

                    train_and_test_features = \
                        build_train_and_test_features(
                            org_df.columns,
                            features_to_process,
                            predict_feature,
                            ignore_features)

                    log.info(("building predict_rows scalers all_features={}")
                             .format(
                                train_and_test_features))

                    scaler_transform_res = \
                        build_scaler_dataset_from_records(
                            label=label,
                            record_list=org_df[
                                train_and_test_features].to_json(),
                            min_feature=min_scaler_range,
                            max_feature=max_scaler_range,
                            cast_to_type=scaler_cast_to_type)

                    if scaler_transform_res["status"] == SUCCESS:
                        log.info(("{} - scaled predict_rows={} "
                                  "df={} dataset={}")
                                 .format(
                            label,
                            len(predict_rows),
                            len(scaler_transform_res["org_recs"].index),
                            len(scaler_transform_res["dataset"])))
                    else:
                        log.error(("{} - failed to scale predict err={} "
                                   "predict_rows={} df={} dataset={}")
                                  .format(
                            label,
                            scaler_transform_res["err"],
                            len(predict_rows),
                            len(scaler_transform_res["org_recs"].index),
                            len(scaler_transform_res["dataset"])))
                    # if scaler works on dataset

                    log.info(("{} building predict org_df scaled "
                              "for testing all samples and predict_rows")
                             .format(
                                label))

                    # noqa https://stackoverflow.com/questions/21764475/scaling-numbers-column-by-column-with-pandas-python
                    row_df = pd.DataFrame(
                                scaler_transform_res["dataset"],
                                columns=org_df[
                                    train_and_test_features].columns)

                    log.info(("{} casting data to floats")
                             .format(
                                label))

                    ml_req = {
                        "X_train": row_df[features_to_process].astype(
                            "float32").values,
                        "Y_train": row_df[predict_feature].astype(
                            "float32").values,
                        "X_test": row_df[features_to_process].astype(
                            "float32").values,
                        "Y_test": row_df[predict_feature].astype(
                            "float32").values
                    }

                    sample_rows = row_df[features_to_process]
                    target_rows = row_df[predict_feature]
                    num_samples = len(sample_rows.index)
                    num_target_rows = len(target_rows.index)
                else:
                    log.info(("{} - no dataset using org_df")
                             .format(
                                label))

                    org_df = pd.read_json(predict_rows)
                    row_df = org_df

                    ml_req = {
                        "X_train": org_df[features_to_process].astype(
                            "float32").values,
                        "Y_train": org_df[predict_feature].astype(
                            "float32").values,
                        "X_test": org_df[features_to_process].astype(
                            "float32").values,
                        "Y_test": org_df[predict_feature].astype(
                            "float32").values
                    }

                    log.info(("{} building predict org_df scaled "
                              "for testing all samples WITHOUT predict_rows")
                             .format(
                                label))

                    log.info(("{} - setting samples "
                              "to features_to_process={} cols={}")
                             .format(
                                label,
                                ppj(features_to_process),
                                list(org_df.columns.values)))
                    sample_rows = row_df[features_to_process]
                    target_rows = row_df[predict_feature]
                    num_samples = len(sample_rows.index)
                    num_target_rows = len(target_rows.index)
                # end of if apply_scalar to csv + predict_rows
            # end of handling metadata-driven split vs controlled

        except Exception as f:
            last_step = ("{} - failed during '{}' json={} "
                         "with ex={}").format(
                            label,
                            last_step,
                            model_json,
                            f)
            log.error(last_step)
            res["status"] = ERR
            res["err"] = last_step
            res["data"] = None
            return res
        # end of try/ex to convert rows to pandas dataframe

        if num_samples == 0:
            last_step = ("{} - "
                         "missing predict_rows={}").format(
                            label,
                            sample_rows)
            log.error(last_step)
            res["status"] = ERR
            res["err"] = last_step
            res["data"] = None
            return res
        # stop if no rows

        # check the headers in the dataframe match what
        # the original model was trained with
        for h in manifest_headers:
            found_header = False
            for d in detected_headers:
                if h == d:
                    found_header = True
                    break
            # end of for all detected headers
            if not found_header:
                last_step = ("{} - invalid predict_rows - header={} in "
                             "detected_headers={}").format(
                        label,
                        h,
                        detected_headers)
                log.error(last_step)
                res["status"] = ERR
                res["err"] = last_step
                res["data"] = None
                return res
        # end for all manifest headers - expected to be in
        # predict_rows

        # create a back up weights file if one was not targeted
        if not weights_file:
            h5_storage_dir = ev(
                "H5_DIR",
                "/tmp")
            weights_file = "{}/{}.h5".format(
                h5_storage_dir,
                str(uuid.uuid4()))
        # end of building a weights file

        last_step = "loading model"
        log.info("{} - {}".format(
            label,
            last_step))

        # load the model from the json
        try:

            if new_model:
                compile_data = {
                    "loss": loss,
                    "optimizer": optimizer,
                    "metrics": metrics
                }

                if ml_type == "standalone-classification":
                    model = build_classification_dnn(
                            num_features=num_features,
                            compile_data=compile_data,
                            label=label,
                            model_json=model_json,
                            model_desc=model_desc)
                elif ml_type == "classification":
                    def set_model():
                        return build_classification_dnn(
                            num_features=num_features,
                            compile_data=compile_data,
                            label=label,
                            model_json=model_json,
                            model_desc=model_desc)

                    model = KerasClassifier(
                            build_fn=set_model,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose)
                elif ml_type == "regression":
                    def set_model():
                        return build_regression_dnn(
                            num_features=num_features,
                            compile_data=compile_data,
                            label=label,
                            model_json=model_json,
                            model_desc=model_desc)

                    model = KerasRegressor(
                            build_fn=set_model,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose)
                else:
                    def set_model():
                        return build_regression_dnn(
                            num_features=num_features,
                            compile_data=compile_data,
                            label=label,
                            model_json=model_json,
                            model_desc=model_desc)

                    model = KerasRegressor(
                            build_fn=set_model,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose)
            else:
                log.info(("{} - using existing - not building={}")
                         .format(
                            label,
                            new_model))
            # end of if new_model or use existing
        except Exception as f:
            last_step = ("{} - failed during '{}' ml_type={} "
                         "model_json={} model_desc={} "
                         "with ex={}").format(
                            label,
                            last_step,
                            ml_type,
                            model_json,
                            model_desc,
                            f)
            log.error(last_step)
            res["status"] = ERR
            res["err"] = last_step
            res["data"] = None
            return res
        # end of try/ex to save weights

        last_step = "saving model weights={}".format(
            weights_file)
        log.info("{} - {}".format(
            label,
            last_step))

        if use_evaluate:
            last_step = ("building training ml_type={} "
                         "predict_type={} "
                         "sample_rows={} target_rows={} "
                         "manifest={}").format(
                            ml_type,
                            predict_type,
                            num_samples,
                            num_target_rows,
                            ppj(manifest))
            log.info("{} - {}".format(
                label,
                last_step))

            # build training request for new predicts

            if not dataset and csv_file:
                ml_req = build_training_request(
                        csv_file=csv_file,
                        meta_file=meta_file,
                        predict_feature=predict_feature,
                        test_size=test_size)

            # fit the model
            if new_model:
                last_step = ("fitting Xtrain={} Ytrain={} Xtest={} Ytest={} "
                             "epochs={} batch_size={}").format(
                                len(ml_req["X_train"]),
                                len(ml_req["Y_train"]),
                                len(ml_req["X_test"]),
                                len(ml_req["Y_test"]),
                                epochs,
                                batch_size)
                log.info("{} - {}".format(
                    label,
                    last_step))
                history = model.fit(
                        ml_req["X_train"],
                        ml_req["Y_train"],
                        validation_data=(
                            ml_req["X_test"],
                            ml_req["Y_test"]),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=verbose)
            else:
                log.info(("{} - using existing - not fitting={}")
                         .format(
                            label,
                            new_model))
            # end of if new model or using existing

        # end of building training update data
        # not use_evaluate

        last_step = ("predicting ml_type={} predict_type={} "
                     "rows={} manifest={}").format(
                        ml_type,
                        predict_type,
                        num_samples,
                        ppj(manifest))
        log.info("{} - {}".format(
            label,
            last_step))

        if os.path.exists(weights_file):
            last_step = ("loading weights_file={}").format(
                            weights_file)
            log.info("{} - {}".format(
                label,
                last_step))

            # load the weights from the file on disk
            try:
                if ml_type == "standalone-classification":
                    model.load_weights(
                        weights_file)
                else:
                    model.model.load_weights(
                        weights_file)
            except Exception as f:
                last_step = ("{} - failed during '{}' "
                             "file={} weights={}"
                             "with ex={}").format(
                                label,
                                last_step,
                                weights_file,
                                weights_json,
                                f)
                log.error(last_step)
                res["status"] = ERR
                res["err"] = last_step
                res["data"] = None
                return res
            # end of try/ex to save weights
        else:
            log.info(("{} did not find weights_file={}")
                     .format(
                        label,
                        weights_file))
        # only load weights if the file is still on disk

        # evaluating
        last_step = ("evaluating num_xtest={} num_ytest={} "
                     "metrics={} histories={} "
                     "loss={} optimizer={}").format(
                        len(ml_req["X_test"]),
                        len(ml_req["Y_test"]),
                        metrics,
                        histories,
                        loss,
                        optimizer)
        log.info("{} - {}".format(
            label,
            last_step))

        # make predictions
        try:
            if should_predict:
                if ml_type == "standalone-classification":
                    if new_model:
                        scores = model.evaluate(
                            ml_req["X_test"],
                            ml_req["Y_test"])
                        if len(scores) > 1:
                            accuracy = {
                                "accuracy": scores[1] * 100
                            }
                        else:
                            accuracy = {
                                "accuracy": 0.0
                            }
                    # no scoring on existing models
                    predictions = model.predict(
                        sample_rows.values,
                        verbose=verbose)
                    numpy.set_printoptions(threshold=numpy.nan)
                    sess = tf.InteractiveSession()  # noqa
                    indexes = tf.argmax(predictions, axis=1)
                    data["indexes"] = indexes
                    rounded = [round(x[0]) for x in predictions]
                    ridx = 0
                    should_set_labels = False
                    labels_dict = {}
                    if "labels" in label_rules \
                       and "label_values" in label_rules:
                        label_rows = label_rules["label_values"]
                        for idx, lidx in enumerate(label_rows):
                            if len(label_rules["labels"]) >= idx:
                                should_set_labels = True
                                labels_dict[str(lidx)] = \
                                    label_rules["labels"][idx]
                    # end of compiling labels dictionary
                    log.info(("{} - scores={} accuracy={} "
                              "merging predictions={} labels={}")
                             .format(
                                label,
                                scores,
                                accuracy.get("accuracy", None),
                                len(sample_rows.index),
                                labels_dict))
                    for idx, row in row_df.iterrows():
                        if len(sample_predictions) > max_records:
                            log.info(("{} hit max={} predictions")
                                     .format(
                                         label,
                                         max_records))
                            break
                        new_row = json.loads(row.to_json())
                        cur_value = rounded[ridx]
                        if predict_feature in row:
                            new_row["_original_{}".format(
                                    predict_feature)] = \
                                row[predict_feature]
                        else:
                            new_row["_original_{}".format(
                                    predict_feature)] = \
                                "missing-from-dataset"
                        new_row[predict_feature] = int(cur_value)
                        if should_set_labels:
                            new_row["label_name"] = \
                                labels_dict[str(int(cur_value))]
                        new_row["_row_idx"] = ridx
                        new_row["_count"] = idx
                        sample_predictions.append(new_row)
                        ridx += 1
                    # end of merging samples with predictions
                elif ml_type == "classification":
                    if new_model:
                        last_step = "building estimators"
                        estimators = []
                        estimators.append(
                            ("standardize",
                             StandardScaler()))
                        estimators.append(
                            ("mlp",
                             model))
                        last_step = "building pipeline"
                        pipeline = Pipeline(estimators)
                        # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/  # noqa
                        last_step = ("{} - starting classification "
                                     "StratifiedKFold "
                                     "splits={} seed={}").format(
                                        label,
                                        num_splits,
                                        seed)
                        log.info(last_step)
                        kfold = StratifiedKFold(
                            n_splits=num_splits,
                            random_state=seed)
                        last_step = "cross_val_score"
                        log.info(("{} - classification cross_val_score: ")
                                 .format(
                                    label))
                        results = cross_val_score(
                            pipeline,
                            ml_req["X_train"],
                            ml_req["Y_train"],
                            cv=kfold)
                        scores = [
                            results.std(),
                            results.mean()
                        ]
                        accuracy = {
                            "accuracy": results.mean() * 100
                        }
                        log.info(("{} - classification accuracy={} samples={}")
                                 .format(
                                    label,
                                    accuracy["accuracy"],
                                    num_samples))
                    else:
                        log.info(("{} - using existing "
                                  "accuracy={} scores={} "
                                  "predictions={}")
                                 .format(
                                    label,
                                    accuracy,
                                    scores,
                                    len(sample_rows.index)))
                    # end of if use existing or new model
                    predictions = model.predict(
                        sample_rows.values,
                        verbose=verbose)
                    if new_model:
                        log.info(("{} - "
                                  "classification confusion_matrix samples={} "
                                  "predictions={} target_rows={}")
                                 .format(
                                    label,
                                    num_samples,
                                    len(predictions),
                                    num_target_rows))
                        cm = confusion_matrix(
                            target_rows.values,
                            predictions)
                        log.info(("{} - "
                                  "classification has confusion_matrix={} "
                                  "predictions={} target_rows={}")
                                 .format(
                                    label,
                                    cm,
                                    len(predictions),
                                    num_target_rows))
                    # end of confusion matrix
                    rounded = [round(x[0]) for x in predictions]
                    ridx = 0
                    should_set_labels = False
                    labels_dict = {}
                    if "labels" in label_rules \
                       and "label_values" in label_rules:
                        label_rows = label_rules["label_values"]
                        for idx, lidx in enumerate(label_rows):
                            if len(label_rules["labels"]) >= idx:
                                should_set_labels = True
                                labels_dict[str(lidx)] = \
                                    label_rules["labels"][idx]
                    # end of compiling labels dictionary
                    log.info(("{} - ml_type={} scores={} accuracy={} "
                              "merging samples={} with predictions={} "
                              "labels={}")
                             .format(
                                label,
                                ml_type,
                                scores,
                                accuracy.get("accuracy", None),
                                len(sample_rows.index),
                                len(rounded),
                                labels_dict))
                    for idx, row in row_df.iterrows():
                        if len(sample_predictions) > max_records:
                            log.info(("{} hit max={} predictions")
                                     .format(
                                         label,
                                         max_records))
                            break
                        new_row = json.loads(row.to_json())
                        cur_value = rounded[ridx]
                        if predict_feature in row:
                            new_row["_original_{}".format(
                                    predict_feature)] = \
                                row[predict_feature]
                        else:
                            new_row["_original_{}".format(
                                    predict_feature)] = \
                                "missing-from-dataset"
                        new_row[predict_feature] = int(cur_value)
                        if should_set_labels:
                            new_row["label_name"] = \
                                labels_dict[str(int(cur_value))]
                        new_row["_row_idx"] = ridx
                        new_row["_count"] = idx
                        sample_predictions.append(new_row)
                        ridx += 1
                    # end of merging samples with predictions
                elif ml_type == "regression":
                    if new_model:
                        last_step = "building new regression model"
                        estimators = []
                        estimators.append(
                            ("standardize",
                             StandardScaler()))
                        estimators.append(
                            ("mlp",
                             model))
                        last_step = "building pipeline"
                        pipeline = Pipeline(estimators)
                        log.info(("{} - starting regression kfolds "
                                  "splits={} seed={}")
                                 .format(
                                    label,
                                    num_splits,
                                    seed))
                        last_step = "starting KFolds"
                        kfold = KFold(
                            n_splits=num_splits,
                            random_state=seed)
                        log.info(("{} - regression cross_val_score "
                                  "kfold={}")
                                 .format(
                                    label,
                                    num_splits))
                        last_step = "starting cross_val_score"
                        results = cross_val_score(
                            pipeline,
                            sample_rows.values,
                            target_rows.values,
                            cv=kfold)
                        log.info(("{} - regression prediction score: "
                                  "mean={} std={}")
                                 .format(
                                    label,
                                    results.mean(),
                                    results.std()))
                        last_step = "getting scores"
                        scores = [
                            results.std(),
                            results.mean()
                        ]
                        last_step = "getting accuracy"
                        accuracy = {
                            "accuracy": results.mean() * 100
                        }
                    # end of if new model or using existing
                    last_step = "making predictions on samples"
                    log.info(("{} - regression accuracy={} samples={}")
                             .format(
                                label,
                                accuracy["accuracy"],
                                num_samples))
                    org_predictions = model.predict(
                        sample_rows.values,
                        verbose=verbose)
                    if apply_scaler and scaler_test:
                        inverse_predictions = scaler_test.inverse_transform(
                            org_predictions.reshape(-1, 1)).reshape(-1)
                        predict_feature_values = \
                            pd.Series(inverse_predictions)
                        inverse_predictions_df = pd.DataFrame(
                            predict_feature_values,
                            columns=[predict_feature])
                        predictions = inverse_predictions_df.values
                        merge_req = {
                            "org_recs": org_df,
                            "inverse_recs": inverse_predictions_df
                        }
                        merge_res = merge_inverse_data_into_original(
                            req=merge_req,
                            sort_on_index=predict_feature,
                            ordered_columns=[predict_feature])

                        merge_df = merge_res["merge_df"]
                        sample_predictions = merge_df.to_json()
                        log.info(("{} - merge_df={}")
                                 .format(
                                    label,
                                    len(sample_predictions)))
                        are_predicts_merged = True
                    else:
                        last_step = "casting predictions to float"
                        predictions = [float(x) for x in org_predictions]

                        log.info(("{} - ml_type={} scores={} accuracy={} "
                                  "merging samples={} with predictions={}")
                                 .format(
                                    label,
                                    ml_type,
                                    scores,
                                    accuracy.get("accuracy", None),
                                    len(sample_rows.index),
                                    len(predictions)))
                        last_step = "merging predictions with org dataframe"
                        ridx = 0
                        for idx, row in row_df.iterrows():
                            if len(sample_predictions) > max_records:
                                log.info(("{} hit max={} predictions")
                                         .format(
                                            label,
                                            max_records))
                                break
                            new_row = json.loads(row.to_json())
                            cur_value = predictions[ridx]
                            if predict_feature in row:
                                new_row["_original_{}".format(
                                        predict_feature)] = \
                                    row[predict_feature]
                            else:
                                new_row["_original_{}".format(
                                        predict_feature)] = \
                                    "missing-from-dataset"
                            new_row[predict_feature] = cur_value
                            new_row["_row_idx"] = ridx
                            new_row["_count"] = idx
                            sample_predictions.append(new_row)
                            log.debug(("predicting={} target={} predicted={}")
                                      .format(
                                        predict_feature,
                                        target_rows[ridx],
                                        new_row[predict_feature]))
                            ridx += 1
                        # end of merging samples with predictions
                    # handle inverse transform for scaler datasets
                    last_step = "merging predictions done"

                else:
                    last_step = ("{} - invalid ml_type={} "
                                 "rows={}").format(
                                    label,
                                    ml_type,
                                    num_samples)
                    res["status"] = ERR
                    res["err"] = last_step
                    res["data"] = None
                    return res
            else:
                log.info(("{} - skipping predictions")
                         .format(
                             should_predict))
        except Exception as f:
            predictions = None
            last_step = ("{} - failed predicting '{}' "
                         "file={} weights={} "
                         "with ex={}").format(
                            label,
                            last_step,
                            weights_file,
                            weights_json,
                            f)
            log.error(last_step)
            res["status"] = ERR
            res["err"] = last_step
            res["data"] = None
            return res
        # end of try/ex to predict

        if ml_type == "classification":
            last_step = ("packaging {} predictions={} "
                         "rows={}").format(
                            ml_type,
                            len(rounded),
                            len(sample_rows.index))
        else:
            last_step = ("packaging {} predictions={} "
                         "rows={}").format(
                            ml_type,
                            len(sample_predictions),
                            len(sample_rows.index))

        log.info(("{} - {}")
                 .format(
                    label,
                    last_step))
        try:
            image_saved = save_prediction_image(
                label=label,
                history=history,
                histories=histories,
                image_file=image_file)
            if image_saved:
                log.info(("{} - created image_file={}")
                         .format(
                             label,
                             image_file))
            else:
                image_file = None
        except Exception as f:
            image_file = None
            last_step = ("{} - failed creating image "
                         "with ex={}").format(
                            label,
                            f)
            log.error(last_step)
        # end of trying to build prediction history image

        model_weights = {}
        if new_model and save_weights:
            try:
                model_weights = {}
                # disabled for https://github.com/keras-team/keras/issues/4875
                # model.save_weights(weights_file)
                # model_weights = model.get_weights()
            except Exception as m:
                log.error(("saving label={} weights_file={} failed ex={}")
                          .format(
                            label,
                            weights_file,
                            m))
            # end of try/ex save
            log.info(("convert label={} weights_file={} to df")
                     .format(
                        label,
                        weights_file))
            try:
                log.info(("label={} weights_file={} to HDFStore")
                         .format(
                            label,
                            weights_file))
                hdf_store = \
                    pd.HDFStore(weights_file)
                log.info(("label={} HDFStore getting keys")
                         .format(
                            label,
                            weights_file))
                hdf_keys = hdf_store.keys()
                model_weights = {
                    "keys": hdf_keys,
                    "weights": []
                }
                log.info(("label={} HDFStore found weight_file={} keys={}")
                         .format(
                            label,
                            weights_file,
                            hdf_keys))
                for cur_key in hdf_keys:
                    log.info(("label={} HDFStore converting keys={} to df")
                             .format(
                                label,
                                cur_key))
                    df_hdf = \
                        pd.read_hdf(weights_file, cur_key)
                    model_weights["weights"].append({
                        "key": cur_key,
                        "df": df_hdf.to_json()})
                # end of for all keys in the HDFStore
            except Exception as m:
                log.error(("convert label={} weights_file={} failed ex={}")
                          .format(
                            label,
                            weights_file,
                            m))
            # end of try/ex convert

        # end of save_weights file for next run

        log.info(("{} - predictions done")
                 .format(
                     label))

        data["acc"] = accuracy
        data["histories"] = histories
        data["image_file"] = image_file
        if new_model:
            if ml_type == "standalone-classification":
                data["model"] = model
            else:
                data["model"] = model.model
        else:
            data["model"] = model

        data["weights"] = model_weights
        data["indexes"] = indexes
        data["scores"] = scores
        data["predictions"] = predictions
        data["rounded"] = rounded
        data["sample_predictions"] = sample_predictions
        data["confusion_matrix"] = None
        data["scaler_train"] = scaler_train
        data["scaler_test"] = scaler_test
        data["scaled_train_dataset"] = scaled_train_dataset
        data["scaled_test_dataset"] = scaled_test_dataset
        data["inverse_predictions"] = inverse_predictions
        data["apply_scaler"] = apply_scaler
        data["merge_df"] = merge_df
        data["are_predicts_merged"] = are_predicts_merged

        res["status"] = SUCCESS
        res["err"] = ""
        res["data"] = data

    except Exception as e:
        res["status"] = ERR
        if existing_model_dict:
            last_step = ("failed {} existing_model request={} hit ex={} "
                         "during last_step='{}'").format(
                            label,
                            req,
                            e,
                            last_step)
        else:
            last_step = ("failed {} request={} hit ex={} "
                         "during last_step='{}'").format(
                            label,
                            ppj(req),
                            e,
                            last_step)
        res["err"] = last_step
        res["data"] = None
        log.error(last_step)
    # end of try/ex

    return res
# end of make_predictions
