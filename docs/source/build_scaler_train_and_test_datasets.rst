Convert Records to Scaler Train and Test Datasets
=================================================

Helper method for converting records into a scaler datasets that are split using ``sklearn.model_selection.train_test_split``
. This means all training and tests data is bounded between a range like: ``[-1, 1]``.

.. automodule:: antinex_utils.build_scaler_train_and_test_datasets
   :members: build_scaler_train_and_test_datasets
