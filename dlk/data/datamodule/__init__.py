# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import abc
import importlib
import os

from lightning import LightningDataModule

from dlk.utils.import_module import import_module_dir


class IBaseDataModule(LightningDataModule):
    """docstring for IBaseDataModule"""

    def __init__(self):
        super(IBaseDataModule, self).__init__()

    def train_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(
            f"You must implementation the train_dataloader for your own datamodule."
        )

    def predict_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(
            f"You must implementation the predict_dataloader for your own datamodule."
        )

    def val_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(
            f"You must implementation the val_dataloader for your own datamodule."
        )

    def test_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(
            f"You must implementation the test_dataloader for your own datamodule."
        )

    @abc.abstractmethod
    def online_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(
            f"You must implementation the online_dataloader for your own datamodule."
        )


# automatically import any Python files in the models directory
datamodule_dir = os.path.dirname(__file__)
import_module_dir(datamodule_dir, "dlk.data.datamodule")
