Make Predictions
================

Large helper method for driving all AI-related tasks.

Handles running:

#.  Building Models
#.  Compiling Models
#.  Creating Datasets for Train, Test, and Predictions
#.  Fitting Models
#.  Evaluating Models
#.  Cross Validating Models
#.  Merging Predictions with Original Records

It looks like this fails to build on readthedocs. Here is the file on GitHub in case the ``automodule`` failed:

`make_predictions.py`_

.. _make_predictions.py: https://github.com/jay-johnson/antinex-utils/blob/master/antinex_utils/make_predictions.py

.. automodule:: antinex_utils.make_predictions
   :members: build_regression_dnn,build_classification_dnn,check_request,save_prediction_image,make_predictions
