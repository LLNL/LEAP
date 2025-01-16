.. _my-filter-sequence-label:

Filter Sequence
===============

The class implements the functionality to specify a list (or just one) of filters/ regularization functionals / denoisers
that can be used in any of the LEAP iterative reconstruction algorithms or can be used to create algorithm just for denoising.
These filters can be either differentiable (e.g., TV) or non-differentiable (e.g. median filter),
but those filters that are not differentiable will be ignored by gradient-based iterative reconstruction algorithms.
Currently the only non-gradient-based iterative reconstruction algorithm in LEAP is ASDPOCS.

For example, if one wishes to employ a Total Variation (TV) regularizer, do the following:

.. code-block:: Python
   
   filters = filterSequence(1.0e0)
   filters.append(TV(leapct, delta=0.02/20.0, p=1.0))

where the argument to filterSequence (in this case 1.0e0) specifies the regularization strength.  Larger values enforce a stronger
regularization penalty and thus produce smoother results.  This parameter should optimized by the user and optimal values
vary on the input data and task.  It is recommended that one first test various powers of ten, e.g., 1e-3, 1e-2, ..., 1e3 and then
narrow ones search once they find something that seems to work pretty well.

For TV denoising, it is important to set the delta parameter to an appropriate value.  See more information about this below in the TV description.

.. autoclass:: leap_filter_sequence.filterSequence
   :members:
.. autoclass:: leap_filter_sequence.denoisingFilter
   :members:
.. autoclass:: leap_filter_sequence.TV
   :members:
.. autoclass:: leap_filter_sequence.BlurFilter
   :members:
.. autoclass:: leap_filter_sequence.BilateralFilter
   :members:
.. autoclass:: leap_filter_sequence.GuidedFilter
   :members:
.. autoclass:: leap_filter_sequence.MedianFilter
   :members:
.. autoclass:: leap_filter_sequence.LpNorm
   :members:
.. autoclass:: leap_filter_sequence.histogramSparsity
   :members:
.. autoclass:: leap_filter_sequence.azimuthalFilter
   :members:
   