import os
import uuid
import json
import numpy
import pandas as pd
import copy
from antinex_utils.log.setup_logging import build_colorized_logger
from antinex_utils.consts import SUCCESS
from antinex_utils.consts import ERR
from antinex_utils.consts import FAILED
from antinex_utils.consts import NOTRUN
from antinex_utils.utils import ev
from antinex_utils.utils import ppj
from antinex_utils.build_training_request import \
    build_training_request
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
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
    if not manifest and not dataset:
        return ("missing manifest "
                "request={}").format(
                    label,
                    ppj(req))
    if manifest:
        csv_file = manifest.get(
            "csv_file",
            None)
        if not predict_rows and not csv_file:
            return ("missing prediction rows or csv_file in "
                    "request={}").format(
                        label,
                        ppj(req))
    if dataset:
        if not predict_rows:
            return ("missing prediction rows for dataset in "
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
        max_records = 100000
        use_evaluate = False
        if csv_file and meta_file and predict_feature:
            if os.path.exists(csv_file) and os.path.exists(meta_file):
                use_evaluate = True
        else:
            if dataset:
                if os.path.exists(dataset):
                    use_evaluate = True

        numpy.random.seed(seed)

        last_step = ("loading prediction "
                     "into dataframe seed={}").format(
                        seed)
        log.info("{} - {}".format(
            label,
            last_step))
        if not weights_file:
            weights_file = manifest.get(
                "model_weights_file",
                None)

        numpy.random.seed(seed)

        last_step = "loading prediction into dataframe"
        log.info("{} - {}".format(
            label,
            last_step))

        # convert json into pandas dataframe for model.predict
        try:
            if new_model and use_evaluate and not predict_rows:
                log.info(("{} - loading predictions from csv={}")
                         .format(
                             label,
                             csv_file))
                org_df = pd.read_csv(csv_file)
                predict_rows = org_df.to_json()
            # end of loading from a csv

            if dataset:
                log.info(("{} loading dataset={}")
                         .format(
                            label,
                            dataset))

                sort_by = req.get(
                    "sort_values",
                    None)
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
                cur_headers = list(org_df.columns.values)
                for h in cur_headers:
                    include_feature = True
                    if h == predict_feature:
                        include_feature = False
                    else:
                        for f in features_to_process:
                            if h == f:
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

                filter_features = copy.deepcopy(features_to_process)
                filter_features.append(predict_feature)

                num_features = len(features_to_process)
                log.info(("{} filtering dataset={} filter_features={}")
                         .format(
                            label,
                            len(org_df.index),
                            ppj(filter_features)))
                filter_df = org_df[filter_features]
                log.info(("{} splitting filtered_dataset={} "
                          "predict_feature={} test_size={} "
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
            # end of handling metadata-driven split vs controlled

            row_df = pd.read_json(predict_rows)
            detected_headers = list(row_df.columns.values)
            log.info(("{} - setting samples "
                      "to features_to_process={} cols={}")
                     .format(
                        label,
                        ppj(features_to_process),
                        list(row_df.columns.values)))
            sample_rows = row_df[features_to_process]
            target_rows = row_df[predict_feature]
            num_samples = len(sample_rows.index)
            num_target_rows = len(target_rows.index)
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
                last_step = ("{} - invalid predict_rows - "
                             "header={} in "
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
                "/tmp/")
            weights_file = "{}/{}".format(
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

            if not dataset:
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
                        estimators = []
                        estimators.append(
                            ("standardize",
                             StandardScaler()))
                        estimators.append(
                            ("mlp",
                             model))
                        pipeline = Pipeline(estimators)
                        # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/  # noqa
                        log.info(("{} - starting classification "
                                  "StratifiedKFold splits={} seed={}")
                                 .format(
                                    label,
                                    num_splits,
                                    seed))
                        kfold = StratifiedKFold(
                            n_splits=num_splits,
                            random_state=seed)
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
                        estimators = []
                        estimators.append(
                            ("standardize",
                             StandardScaler()))
                        estimators.append(
                            ("mlp",
                             model))
                        pipeline = Pipeline(estimators)
                        log.info(("{} - starting regression kfolds "
                                  "splits={} seed={}")
                                 .format(
                                    label,
                                    num_splits,
                                    seed))
                        kfold = KFold(
                            n_splits=num_splits,
                            random_state=seed)
                        log.info(("{} - regression cross_val_score "
                                  "kfold={}")
                                 .format(
                                    label,
                                    num_splits))
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
                        scores = [
                            results.std(),
                            results.mean()
                        ]
                        accuracy = {
                            "accuracy": results.mean() * 100
                        }
                    # end of if new model or using existing
                    log.info(("{} - regression accuracy={} samples={}")
                             .format(
                                label,
                                accuracy["accuracy"],
                                num_samples))
                    org_predictions = model.predict(
                        sample_rows.values,
                        verbose=verbose)
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

        res["status"] = SUCCESS
        res["err"] = ""
        res["data"] = data

    except Exception as e:
        res["status"] = ERR
        last_step = ("failed {} predictions "
                     "ex={} request={}").format(
                        label,
                        e,
                        ppj(req))
        res["err"] = last_step
        res["data"] = None
        log.error(last_step)
    # end of try/ex

    return res
# end of make_predictions
