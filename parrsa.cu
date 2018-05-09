#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long ulint;
typedef unsigned long long ulint64;

int banyakdata = 1024;
int dimensigrid = 8;
int dimensiblok = 128;

__device__ void modexp(ulint a, ulint b, ulint c, ulint* res) {
	ulint64 s = a;
	ulint64 ans = 1;
	while (b != 0) {
		if (b % 2 == 1) {
			ans = ans * s % c;
			b--;
		}
		b /= 2;
		if (b != 0) {
			s = s * s %c;
		}
	}
	*res = ans;
}

__global__ void kernelenk(ulint *m, ulint e, ulint n, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	modexp(m[i], e, n, res + i);
}

__global__ void kerneldek(ulint *c, ulint d, ulint n, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	modexp(c[i], d, n, res + i);
}

void enkripsiCUDA(ulint *m, ulint e, ulint n, ulint *res) {
	cudaSetDevice(0);

	ulint *devm, *devres;

	cudaMalloc((void**)&devm, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres, banyakdata * sizeof(ulint));
	
	cudaMemcpy((devm), m, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernelenk << <dimensigrid, dimensiblok>> >(devm,e,n,devres);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nDurasi enkripsi= %f ms\n", milliseconds);


	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res, devres, (sizeof(ulint) * banyakdata), cudaMemcpyDeviceToHost);
	
	cudaFree(devm);
	cudaFree(devres);
}

void dekripsiCUDA(ulint *c, ulint d, ulint n, ulint *res2) {
	cudaSetDevice(0);

	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devc, *devres2;
	

	cudaMalloc((void**)&devc, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres2, banyakdata * sizeof(ulint));
	
	cudaMemcpy((devc), c, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);

	// printf("<<<<<<<<<<<<<<<<<<KERNEL>>>>>>>>>>>>>>>>>\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kerneldek << <dimensigrid, dimensiblok>> >(devc,d,n,devres2);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nDurasi enkripsi= %f ms\n", milliseconds);

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res2, devres2, (sizeof(ulint) * banyakdata), cudaMemcpyDeviceToHost);
	
	cudaFree(devc);
	cudaFree(devres2);

}

void initenkripsi(ulint *m){
	srand(2018);
	for (int i = 0; i < banyakdata; i++) {
		m[i] = rand() % 256;
	}	
}

int main(){
	ulint *m, e, d, n, *res, *res2;

	m = (ulint*)malloc(banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * sizeof(ulint));
	res2 = (ulint*)malloc(banyakdata * sizeof(ulint));

	e = 211;
	d = 259;
	n = 299;

	initenkripsi(m);

	printf("<<<<<<<<<<<<<<Pesan Asli>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("m[%d] = %lu\n", i, m[i]);
	}

	printf("m[...]\n");
	printf("m[%d] = %lu\n", banyakdata-1, m[banyakdata-1]);

	enkripsiCUDA(m,e,n,res);

	printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("c[%d] = %lu 	c[%d] = %lu\n", 2*i, res[2*i], 2*i+1, res[2*i+1]);
	}

	printf("c ...\n");
	printf("c[%d] = %lu 	c[%d] = %lu\n", banyakdata * 2-2, res[banyakdata * 2-2], banyakdata *2-1,res[banyakdata*2-1]);

	dekripsiCUDA(res,d,n,res2);

	printf("<<<<<<<<<<<<<<Hasil Dekripsi>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("m[%d] = %lu\n", i, res2[i]);
	}

	printf("m[...]\n");
	printf("m[%d] = %lu\n", banyakdata-1, res2[banyakdata-1]);

	free(m);
	free(res);
	free(res2);

	return 0;
}