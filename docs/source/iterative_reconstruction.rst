Iterative Reconstruction
========================

This page describes the iterative reconstruction algorithm in LEAP.  These algorithms can be applied to any CT geometry.
Iterative reconstruction algorithms may outperform Filtered Backprojection (FBP) reconstruction when
the reconstruction problem is severly ill-posed such as very noisey data, not enough projections,
or projections that only cover a limited angular range.  Sometimes iterative reconstruction can underperform
FBP due to overfitting when the line integral model is not an accurate model.  Inaccurate models can be the result of
incorrect CT geometry specification, strong beam hardening or scatter effects, etc.

We shall separate the following iterative reconstruction algorithms into the following categories

* Algebraic: These algorithms use algebraic means to solve the reconstruction problem.  The time per iteration is fast, but the algorithms are pretty basic.
   * SIRT
   * SART

* Statistical (Transmission): These algorithm model the statistics of transmission CT data and are very good at reconstructing noisey data.
   * RWLS
   * MLTR

* Statistical (Emission): These algorithm model the statistics of emission CT data and are very good at reconstructing noisey data.
   * MLEM
   * OSEM

* Special Purpose: These algorithms are good at solving reconstruction problems where one has a small number of projections and/or the projections only cover a limited angular range.
   * RDLS
   * ASDPOCS


We shall use the following notation in this section

* :math:`g`: projection (attenuation) data

* :math:`f`: reconstruction volume

* :math:`P`: forward projection operator

* :math:`P^T`: backprojection operator (adjoint of forward projection)

* :math:`\boldsymbol{1}`: vector of all ones

* :math:`R(\cdot)`: regularization functional (e.g., TV)

For more information on how to specify regularization functions in LEAP, see the :ref:`my-filter-sequence-label` description.

.. autofunction:: leapctype.tomographicModels.SIRT
.. autofunction:: leapctype.tomographicModels.SART
.. autofunction:: leapctype.tomographicModels.ASDPOCS
.. autofunction:: leapctype.tomographicModels.LS
.. autofunction:: leapctype.tomographicModels.WLS
.. autofunction:: leapctype.tomographicModels.RLS
.. autofunction:: leapctype.tomographicModels.RWLS
.. autofunction:: leapctype.tomographicModels.DLS
.. autofunction:: leapctype.tomographicModels.RDLS
.. autofunction:: leapctype.tomographicModels.MLTR
.. autofunction:: leapctype.tomographicModels.MLEM
.. autofunction:: leapctype.tomographicModels.OSEM
