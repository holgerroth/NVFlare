#! /usr/bin/env python3
## ---------------------------------------------------------------------------
##
## File: newton_raphson_persistor.py for Newton Raphson
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Wed Mar 20 14:16:54 2024 Zhijin Li
## Last update Mon Apr 15 00:39:22 2024 Zhijin Li
## ---------------------------------------------------------------------------


import os

import numpy as np

from nvflare.app_common.np.np_model_persistor import NPModelPersistor


class NewtonRaphsonModelPersistor(NPModelPersistor):
  """
  This class defines the persistor for Newton Raphson model.

A persistor controls the logic behind initializing, loading
  and saving of the model / parameters for each round of a
  federated learning process.

  In the 2nd order Newton Raphson case, a model is just a
  1-D numpy vector containing the parameters for logistic
  regression. The length of the parameter vector is defined
  by the number of features in the dataset.

  """

  def __init__(
      self,
      model_dir="models",
      model_name="weights.npy",
      n_features=13
  ):
    """
    Init function for NewtonRaphsonModelPersistor.

    Args:
        model_dir: sub-folder name to save and load the global model
            between rounds.
        model_name: name to save and load the global model.
        n_features: number of features for the logistic regression.
            For the UCI ML heart Disease dataset, this is 13.

    """

    super().__init__()

    self.model_dir = model_dir
    self.model_name = model_name
    self.n_features = n_features

    # A default model is loaded when no local model is available.
    # This happen when training starts.
    #
    # A `model` for a binary logistic regression is just a matrix,
    # with shape (n_features + 1, 1).
    # For the UCI ML Heart Disease dataset, the n_features = 13.
    #
    # A default matrix with value 0s is created.
    #
    self.default_data = np.zeros(
      (self.n_features + 1, 1),
      dtype=np.float32
    )
