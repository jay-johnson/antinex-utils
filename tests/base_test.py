import os
import json
import uuid
import unittest
import pandas as pd
from antinex_utils.utils import ppj
from antinex_utils.consts import SUCCESS
from antinex_utils.build_scaler_dataset_from_records import \
    build_scaler_dataset_from_records


class BaseTestCase(unittest.TestCase):

    def build_model_and_weights(
            self,
            data_file=("./tests/keras/"
                       "model_and_weights.json")):
        """build_model_and_weights

        :param data_file: file with model and weights
        """
        file_contents = open(data_file).read()
        data = json.loads(file_contents)
        return data
    # end of build_model_and_weights

    def build_prediction_rows(
            self,
            data_file=("./tests/datasets/classification/"
                       "cleaned_attack_scans.csv")):
        """build_prediction_rows

        :param data_file: file with model and weights
        """
        csv_data = pd.read_csv(data_file)
        data = csv_data.to_json()
        return data
    # end of build_prediction_rows

    def build_classification_request(
            self,
            data_file=("./tests/datasets/classification/"
                       "classification_request.json"),
            predict_rows_file=("./tests/datasets/classification/"
                               "cleaned_attack_scans.csv"),
            meta_file=("./tests/datasets/classification/"
                       "cleaned_metadata.json"),
            model_desc_file=("./tests/model_desc/"
                             "simple_dnn.json"),
            model_json_file=("./tests/keras/"
                             "model_and_weights.json"),
            model_weights_file=None):

        predict_rows = self.build_prediction_rows(
            data_file=predict_rows_file)

        predict_manifest = None
        model_desc = None
        model_json = None
        meta_json = None
        with open(data_file) as cur_file:
            predict_manifest = json.loads(cur_file.read())

        with open(model_desc_file) as cur_file:
            model_desc = json.loads(cur_file.read())

        with open(model_json_file) as cur_file:
            model_json = json.loads(cur_file.read())

        with open(meta_file) as cur_file:
            meta_json = json.loads(cur_file.read())

        # set the classification inputs
        model_json["model"]["config"][0]["config"]["batch_input_shape"][1] = \
            len(predict_manifest["manifest"]["features_to_process"])

        prediction_req = {
            "label": "testing_{}".format(
                str(uuid.uuid4())),
            "predict_rows": predict_rows,
            "model_desc": model_desc,
            "manifest": predict_manifest["manifest"],
            "model_json": model_json["model"],
            "weights_json": model_json["weights"],
            "meta": meta_json
        }

        return prediction_req
    # end of build_classification_request

    def build_regression_request(
            self,
            data_file=("./tests/datasets/regression/"
                       "regression_request.json"),
            predict_rows_file=("./tests/datasets/regression/"
                               "stock.csv"),
            meta_file=("./tests/datasets/regression/"
                       "stock_metadata.json"),
            model_desc_file=("./tests/model_desc/"
                             "simple_dnn.json"),
            model_json_file=("./tests/keras/"
                             "model_and_weights.json"),
            model_weights_file=None):

        predict_rows = self.build_prediction_rows(
            data_file=predict_rows_file)

        predict_manifest = None
        model_desc = None
        model_json = None
        meta_json = None
        with open(data_file) as cur_file:
            predict_manifest = json.loads(cur_file.read())

        with open(model_desc_file) as cur_file:
            model_desc = json.loads(cur_file.read())

        with open(model_json_file) as cur_file:
            model_json = json.loads(cur_file.read())

        with open(meta_file) as cur_file:
            meta_json = json.loads(cur_file.read())

        # set the regression inputs
        model_json["model"]["config"][0]["config"]["batch_input_shape"][1] = \
            len(predict_manifest["manifest"]["features_to_process"])

        prediction_req = {
            "label": "testing_{}".format(
                str(uuid.uuid4())),
            "predict_rows": predict_rows,
            "model_desc": model_desc,
            "manifest": predict_manifest["manifest"],
            "model_json": model_json["model"],
            "weights_json": model_json["weights"],
            "meta": meta_json
        }

        if bool(os.getenv("TEST_DEBUG", "0") == "1"):
            print(ppj(prediction_req))

        return prediction_req
    # end of build_regression_request

    def build_dataset_regression_request(
            self,
            data_file=("./tests/datasets/regression/"
                       "dataset_prediction.json"),
            predict_rows_file=("./tests/datasets/regression/"
                               "stock.csv")):

        predict_rows = self.build_prediction_rows(
            data_file=predict_rows_file)

        dataset_manifest = None
        with open(data_file) as cur_file:
            dataset_manifest = json.loads(cur_file.read())

        prediction_req = dataset_manifest
        prediction_req["label"] = "testing_{}".format(
                str(uuid.uuid4()))
        prediction_req["predict_rows"] = predict_rows

        if bool(os.getenv("TEST_DEBUG", "0") == "1"):
            print(ppj(prediction_req))

        return prediction_req
    # end of build_dataset_regression_request

    def build_dataset_regression_with_scaler(
            self,
            predict_rows_file=("./tests/datasets/regression/"
                               "stock.csv")):

        predict_rows = pd.read_csv(
                    predict_rows_file).to_json()

        res = build_scaler_dataset_from_records(
            record_list=predict_rows)

        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["scaler"] is not None)
        self.assertTrue(
            res["dataset"] is not None)
        self.assertTrue(
            res["org_recs"] is not None)

        dataset = res["dataset"]
        scaler = res["scaler"]
        res["inverse_recs"] = scaler.inverse_transform(dataset)

        return res
    # end of build_dataset_regression_with_scaler

# end of BaseTestCase
