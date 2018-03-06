from tests.base_test import BaseTestCase
from antinex_utils.make_predictions import make_predictions
from antinex_utils.consts import SUCCESS


class TestRegression(BaseTestCase):

    def test_regression(self):
        req = self.build_regression_request()
        print(req)
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

# end of TestRegression
