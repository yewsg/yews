yews.transforms
======================

.. currentmodule:: yews.transforms

Transforms are common waveform transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`yews.transforms.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline.

.. autoclass:: Compose

Transforms on Numpy Array
-------------------------

.. autoclass:: ZeroMean

.. autoclass:: SoftClip

.. autoclass:: CutWaveform

Conversion Transforms
---------------------

.. autoclass:: ToTensor
	:members: __call__
	:special-members:

Functional Transforms
---------------------

Functional transforms give you fine-grained control of the transformation pipeline.
As opposed to the transformations above, functional transforms don't contain a random number
generator for their parameters.
That means you have to specify/generate all parameters, but you can reuse the functional transform.
For example, you can apply a functional transform to multiple images like this:

TODO: need to replace the image example by a seismic waveform example below:

.. code:: python

    import yews.transforms.functional as TF
    import random

    def my_segmentation_transforms(image, segmentation):
        if random.random() > 5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        # more transforms ...
        return image, segmentation

.. automodule:: yews.transforms.functional
   :members:

