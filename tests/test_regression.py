import json
from tests.base_test import BaseTestCase
from antinex_utils.make_predictions import make_predictions
from antinex_utils.merge_inverse_data_into_original import \
    merge_inverse_data_into_original
from antinex_utils.consts import SUCCESS


class TestRegression(BaseTestCase):

    def test_regression(self):
        req = self.build_regression_request()
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_regression

    def test_regression_wide_dnn(self):
        req = self.build_regression_request(
            model_desc_file="./tests/model_desc/wide_dnn.json")
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_regression_wide_dnn

    def test_regression_simple_dnn(self):
        req = self.build_regression_request(
            model_desc_file="./tests/model_desc/simple_dnn.json")
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_regression_simple_dnn

    def test_regression_deep_dnn(self):
        req = self.build_regression_request(
            model_desc_file="./tests/model_desc/deep_dnn.json")
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_regression_deep_dnn

    def test_dataset_regression(self):
        req = self.build_dataset_regression_request()
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
        self.assertTrue(
            len(res["data"]["model"].model.layers) == 4)
    # end of test_dataset_regression

    def test_dataset_regression_scaler_utils(self):
        """test_dataset_regression_scaler_utils"""
        req = self.build_dataset_regression_with_scaler()
        ordered_columns = list(req["org_recs"].columns.values)
        sort_on_index = ordered_columns[0]
        predict_feature = "close"

        res = merge_inverse_data_into_original(
            req=req,
            sort_on_index=sort_on_index,
            ordered_columns=ordered_columns)
        self.assertEqual(
            res["status"],
            SUCCESS)

        sorted_org_df = res["sorted_org_df"]
        predict_df = res["predict_df"]

        predict_df.set_index(sort_on_index)
        row_idx = 0
        for idx, predict_row in predict_df.iterrows():
            org_row_dict = json.loads(
                sorted_org_df.iloc[row_idx].to_json())
            predict_dict = json.loads(predict_row.to_json())
            for key in org_row_dict:
                self.assertEqual(
                    org_row_dict[key],
                    predict_dict[key])
            self.assertTrue(
                org_row_dict.get(
                    predict_feature,
                    True))
            self.assertTrue(
                predict_dict.get(
                    predict_feature,
                    False))
            row_idx += 1
        # end of for all rows to check for ordering

    # end of test_dataset_regression_scaler_utils

    def test_dataset_regression_using_scaler(self):
        req = self.build_dataset_regression_request()
        req["apply_scaler"] = True
        req["scaler_cast_type"] = "float32"
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
        self.assertTrue(
            len(res["data"]["model"].model.layers) == 4)

        predictions = res["data"]["sample_predictions"]
        self.assertTrue(
            len(predictions),
            18)
    # end of test_dataset_regression_using_scaler

    def test_regression_wide_dnn_with_auto_scaler(self):
        req = self.build_regression_request(
            model_desc_file="./tests/model_desc/wide_dnn.json")
        req["apply_scaler"] = True
        req["scaler_cast_type"] = "float32"
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_regression_wide_dnn_with_auto_scaler

# end of TestRegression
