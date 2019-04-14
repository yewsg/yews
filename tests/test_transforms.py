import pytest
import yews.transforms as transforms
import yews.transforms.functional as F

import torch
import numpy as np

class TestIsNumpyWaveform:
    def test_single_channel_waveform_vector(self):
        wav = np.empty(10)
        assert F._is_numpy_waveform(wav)

    def test_single_channel_waveform_matrix(self):
        wav = np.empty((1, 10))
        assert F._is_numpy_waveform(wav)

    def test_multi_channel_waveform(self):
        wav = np.empty((3, 10))
        assert F._is_numpy_waveform(wav)

    def test_invalid_waveform_wrong_dimension(self):
        wav = np.empty((1,1,10))
        assert not F._is_numpy_waveform(wav)

    def test_invalid_waveform_wrong_type(self):
        wav = torch.tensor(10)
        assert not F._is_numpy_waveform(wav)

class TestToTensor:
    def test_type_exception(self):
        wav = torch.tensor(10)
        with pytest.raises(TypeError):
            F._to_tensor(wav)

    def test_single_channel_waveform(self):
        wav = np.zeros(10)
        tensor = torch.zeros(1, 10,dtype=torch.float)
        assert torch.allclose(F._to_tensor(wav), tensor)

    def test_multi_channel_waveform(self):
        wav = np.zeros((3, 10))
        tensor = torch.zeros(3, 10,dtype=torch.float)
        assert torch.allclose(F._to_tensor(wav), tensor)

class TestMandatoryMethods:
    def test_call_method(self):
        assert all([hasattr(getattr(transforms, t), '__call__') for t in
                    transforms.transforms.__all__])

    def test_repr_method(self):
        assert all([hasattr(getattr(transforms, t), '__repr__') for t in
                    transforms.transforms.__all__])

class TestTransformCorrectness:
    def test_compose(self):
        wav = np.array([1, 3])
        assert torch.allclose(transforms.Compose([
            transforms.ZeroMean(),
            transforms.ToTensor(),
        ])(wav), torch.tensor([[-1, 1]], dtype=torch.float))

    def test_to_tensor_dtype(self):
        wav = np.array([1])
        assert torch.allclose(torch.tensor([[1]], dtype=torch.float),
                                transforms.ToTensor()(wav))
    def test_to_tensor_shape(self):
        wav = np.array([1])
        assert transforms.ToTensor()(wav).shape == (1,1)
