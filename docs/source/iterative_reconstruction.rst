Iterative Reconstruction
========================

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
