# Convolve

Bare correlation kernels (NaN-naive) and their NaN-as-missing
renormalized variants, plus a Gaussian kernel constructor.

The kernels are written as **correlation** (the kernel is not flipped).
For a symmetric kernel correlation and convolution coincide; the
`image.convolve_psf` wrapper flips the PSF first to give true
convolution.

## Gaussian Kernel

::: noobase._core.convolve.gaussian1d

## 1-D Correlation

::: noobase._core.convolve.conv1d

::: noobase._core.convolve.conv_axis

## 2-D Correlation

::: noobase._core.convolve.conv2d

## NaN-as-Missing Renormalized Correlation

::: noobase._core.convolve.conv2d_renorm

::: noobase._core.convolve.conv_axis_renorm
