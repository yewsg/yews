import pytest
import torch
from yews import models


class TestCPICModel:

    def test_cpic_model(self):
        dummpy_input = torch.randn(1, 3, 2000)
        model = models.cpic()
        model(dummpy_input)
