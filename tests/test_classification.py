from tests.base_test import BaseTestCase
from antinex_utils.make_predictions import make_predictions
from antinex_utils.consts import SUCCESS


class TestClassification(BaseTestCase):

    def test_classification(self):
        req = self.build_classification_request()
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_classification

    def test_classification_wide_dnn(self):
        req = self.build_classification_request(
            model_desc_file="./tests/model_desc/wide_dnn.json")
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_classification_wide_dnn

    def test_classification_simple_dnn(self):
        req = self.build_classification_request(
            model_desc_file="./tests/model_desc/simple_dnn.json")
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_classification_simple_dnn

    def test_classification_deep_dnn(self):
        req = self.build_classification_request(
            model_desc_file="./tests/model_desc/deep_dnn.json")
        res = make_predictions(req)
        self.assertEqual(
            res["status"],
            SUCCESS)
        self.assertTrue(
            res["data"]["model"])
    # end of test_classification_deep_dnn

# end of TestClassification
