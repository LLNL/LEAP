#include "cuda_utils.h"

dim3 setBlockSize(int3 N)
{
	dim3 dimBlock(8, 8, 8);  // needs to be optimized
	if (N.z < 8)
	{
		dimBlock.x = 16;
		dimBlock.y = 16;
		dimBlock.z = 1;
	}
	else if (N.y < 8)
	{
		dimBlock.x = 16;
		dimBlock.y = 1;
		dimBlock.z = 16;
	}
	else if (N.x < 8)
	{
		dimBlock.x = 1;
		dimBlock.y = 16;
		dimBlock.z = 16;
	}
	return dimBlock;
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N.x = N_txt.z;
        N.z = N_txt.x;
    }
    return loadTexture(tex_object, dev_data, N, useExtrapolation, useLinearInterpolation);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N.x = N_txt.z;
        N.z = N_txt.x;
    }
    return loadTexture(tex_object, dev_data, N, useExtrapolation, useLinearInterpolation);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    int3 N3 = make_int3(N_txt.x, N_txt.y, N_txt.z);
    return loadTexture(tex_object, dev_data, N3, useExtrapolation, useLinearInterpolation);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
  if (dev_data == nullptr)
    return nullptr;
  cudaArray* d_data_array = nullptr;

  // Allocate 3D array memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&d_data_array, &channelDesc, make_cudaExtent(N_txt.z, N_txt.y, N_txt.x));
 
  // Bind 3D array to texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = (cudaArray_t)d_data_array;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = false;  // Texture coordinates normalization

  if (useExtrapolation)
  {
    texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeClamp;
    texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeClamp;
    texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeClamp;
  }
  else
  {
    texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeBorder;
    texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeBorder;
    texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeBorder;
  }

  if (useLinearInterpolation)
  {
      texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModeLinear;
  }
  else
  {
      texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModePoint;
  }
  cudaCreateTextureObject(&tex_object, &resDesc, &texDesc, nullptr);

  // Update the texture memory
  cudaMemcpy3DParms cudaparams = {0};
  cudaparams.extent = make_cudaExtent(N_txt.z, N_txt.y, N_txt.x);
  cudaparams.kind = cudaMemcpyDeviceToDevice;
  cudaparams.srcPos = make_cudaPos(0, 0, 0);
  cudaparams.srcPtr = make_cudaPitchedPtr(dev_data, N_txt.z * sizeof(float), N_txt.z, N_txt.y);
  cudaparams.dstPos = make_cudaPos(0, 0, 0);
  cudaparams.dstArray = (cudaArray_t)d_data_array;
  cudaMemcpy3D(&cudaparams);
  return d_data_array;
}

float* copyProjectionDataToGPU(float* g, parameters* params, int whichGPU)
{
	cudaSetDevice(whichGPU);

	int N = params->numAngles * params->numRows * params->numCols;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_g = 0;
	if (cudaMalloc((void**)&dev_g, N * sizeof(float)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(projection) failed!\n");
		return NULL;
	}
	if (cudaMemcpy(dev_g, g, N * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy(projection) failed!\n");
		return NULL;
	}

	return dev_g;
}

bool pullProjectionDataFromGPU(float* g, parameters* params, float* dev_g, int whichGPU)
{
	cudaSetDevice(whichGPU);
	cudaError_t cudaStatus;

	int N = params->numAngles * params->numRows * params->numCols;

	cudaStatus = cudaMemcpy(g, dev_g, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy projection data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copyVolumeDataToGPU(float* f, parameters* params, int whichGPU)
{
	cudaSetDevice(whichGPU);

	int N = params->numX * params->numY * params->numZ;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_f = 0;
	if (cudaMalloc((void**)&dev_f, N * sizeof(float)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(volume) failed!\n");
		return NULL;
	}
	if (cudaMemcpy(dev_f, f, N * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy(volume) failed!\n");
		return NULL;
	}

	return dev_f;
}

bool pullVolumeDataFromGPU(float* f, parameters* params, float* dev_f, int whichGPU)
{
	cudaSetDevice(whichGPU);
	cudaError_t cudaStatus;
	int N = params->numX * params->numY * params->numZ;
	cudaStatus = cudaMemcpy(f, dev_f, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy volume data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copy3DdataToGPU(float* g, int3 N, int whichGPU)
{
	cudaSetDevice(whichGPU);

	int N_prod = N.x * N.y * N.z;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_g = 0;
	if (cudaMalloc((void**)&dev_g, N_prod * sizeof(float)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(volume) failed!\n");
		return NULL;
	}
	if (cudaMemcpy(dev_g, g, N_prod * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy(volume) failed!\n");
		return NULL;
	}

	return dev_g;
}

bool pull3DdataFromGPU(float* g, int3 N, float* dev_g, int whichGPU)
{
	cudaSetDevice(whichGPU);
	cudaError_t cudaStatus;
	int N_prod = N.x * N.y * N.z;
	cudaStatus = cudaMemcpy(g, dev_g, N_prod * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy volume data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

