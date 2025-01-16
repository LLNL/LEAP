# Noise and Resolution Properties of Images Reconstructed with Voxel-Driven (VD) and Separable-Footprint (SF) Backprojector Models

# Introduction

LEAP is capable of performing SF-based forward and backprojection and VD-based backprojection for all CT geometries.  To switch between the different backprojection methods, use the following commands:
```
leapct.set_projector('SF') # this is the default setting
leapct.set_projector('VD')
```

Briefly, SF projectors are more accurate, but VD backprojection is faster.  For example, VD backprojection of cone-beam data is about twice as fast as SF-based backprojection in LEAP.  Here we shall estimate the noise and resolution properties of these two methods.  The results of these tests can be re-generated using the [d98_SF_vs_VD.py](https://github.com/LLNL/LEAP/blob/main/demo_leapctype/d98_SF_vs_VD.py) script.

Resolution of CT systems is largely determined by the detector pixel sizes.  For cone-beam systems, one can get high image resolution by taking advantage of geometric magnification.  Geometric magnification increases (and thus resolution improves) as the sample is placed closer to the x-ray source.  One must be careful not to place the object too close to the x-ray source where either the object extends past the field of view of the system or one incurs resolution loss from the size of the x-ray source.  Assuming the the x-ray source spot size does not play a factor in resolution, then one may approximate the system resolution as 
```
sod/sdd*pixelWidth
```
where sod is the source to object distance, sdd is the source to detector distance, and pixelWidth is the detector pixel pitch.  We shall call this quantity the "nominal" system resolution or "nominal" voxel size.

The concept of voxel-driven backprojection is simple.  For each source and detector position, one calculates the position of the line from the source, through the center of a voxel of interest hits the detector.  Then the backprojection at this source and detector position is calculated by bilinear interpolation of where the line described in the previous sentence hits the detector.  That's it.  Other than a variable scaling, the interpolation parameters do not depend on the size of the voxel or its distance from the source and detector.  Thus, small, medium, and large voxel sizes all use the same intepolation parameters.

Methods such as Separable Footprint take into account the voxel and detector pixel sizes and the voxel's distance from the source and detector.  Here, one calculates where the edges of each voxel hit the detector which outlines a rectangular region on the detector that models the 3D projection of the voxel onto the detector.  We shall call this the voxel footprint.  Then the backprojection is calculated by summing up the products of the detector pixel value the area of intersection of the voxel footprint and the detector area for all detector pixels with non-zero intersection with the voxel footprint.  This is a much more accurate model of the x-ray absorption effect in practice.  The consequence of this is that smaller voxels or those closer to the detector have a smaller footprint on the detector, and conversely, larger voxels or those that are further from the detector have a larger footprint on the detector.

If one aims to get a high resolution reconstruction, SF projectors are much more effective because one can use small voxels whose footprint spans 1-4 detector pixels, depending on their position.  The backprojection will only include those detector pixels that are in the shadow of the small voxel.  On the other hand, VD backprojection will always use bilinear interpolation across four detector pixels which will effectively lower the resolution of the reconstruction.

If one aims to get a lower noise reconstruction, SF projectors can be more effective by using larger voxels whose footprint spans many detectors.  As the voxel footprint spans more detector pixels, more averaging will be performed which will reduce noise.  On the other hand, as stated above, VD backprojection will always use bilinear interpolation across four detector pixels which will only average the backprojection over four detector pixels.  This is the reason why the LEAP result of the walnut reconstruction on the LEAP readme page had higher signal to noise ratio.

We shall demonsrate these claims in the sections below.

# Voxels that are smaller than the nominal size

Here we used a voxel size equal to
```
0.5 * sod / sdd * pixelWidth
```
and performed an FBP reconstruction using VD- and SF-based backprojection.  The modulation transfer function (MTF) of these reconstructions is plotted below.
<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/results/MTF_0.png>
</p>
Indeed, just as predicted, the SF result has high resolution.

# Voxels that are the nominal size

Here we used a voxel size equal to
```
sod / sdd * pixelWidth
```
and performed an FBP reconstruction using VD- and SF-based backprojection.  The modulation transfer function (MTF) of these reconstructions is plotted below.
<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/results/MTF_1.png>
</p>
Indeed, just as predicted, the SF and VD backprojectors produce nearly equivalent resolution.

# Voxels that are larger than the nominal size

Here we used a voxel size equal to
```
2.0 * sod / sdd * pixelWidth
```
and performed an FBP reconstruction using VD- and SF-based backprojection.  The modulation transfer function (MTF) of these reconstructions is plotted below.
<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/results/MTF_2.png>
</p>
And here are the reconstructed slices when noise was added to the projection data
<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/results/noisey_2.png>
</p>
Indeed, just as predicted, the VD result has high resolution, but the SF result has higher SNR.  The SNR for the VD result is 25.7 and the SNR for the SF result is 43.6.
