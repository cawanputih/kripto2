#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef long ulint;
typedef long long ulint64;

int banyakdata = 1024;
int dimensigrid = 8;
int dimensiblok = 128;

__device__ ulint modexp(ulint a, ulint b, ulint c) {
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
	return ans;
}

__device__ void enkripsi(ulint m, ulint n, ulint *res) {
	*res = m*m % n;
}

__device__ void dekripsi(ulint c, ulint p, ulint q, ulint pi, ulint qi, ulint n, ulint *res) {
	ulint mp = modexp(c, (p+1)/4, p);
	ulint mq = modexp(c, (q+1)/4, q);

	*res = (pi * p * mq + qi * q * mp) % n;
	*(res+1) = (pi * p * mq - qi * q * mp) % n;
}

__global__ void kernelenk(ulint *m, ulint n, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	enkripsi(m[i], n, res + i);
}

__global__ void kerneldek(ulint *c, ulint p, ulint q, ulint pi, ulint qi, ulint n, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	dekripsi(c[i], p, q, pi, qi, n, res + 2*i);
}

void enkripsiCUDA(ulint *m, ulint n, ulint *res) {
	cudaSetDevice(0);

	ulint *devm, *devres;

	cudaMalloc((void**)&devm, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres, banyakdata * sizeof(ulint));
	
	cudaMemcpy((devm), m, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernelenk << <dimensigrid, dimensiblok>> >(devm,n,devres);

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

void dekripsiCUDA(ulint *c, ulint p, ulint q, ulint pi, ulint qi, ulint n, ulint *res2) {
	cudaSetDevice(0);

	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devc, *devres2;
	

	cudaMalloc((void**)&devc, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres2, banyakdata * 2 * sizeof(ulint));
	
	cudaMemcpy((devc), c, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);

	// printf("<<<<<<<<<<<<<<<<<<KERNEL>>>>>>>>>>>>>>>>>\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kerneldek << <dimensigrid, dimensiblok>> >(devc,p,q,pi,qi,n,devres2);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nDurasi enkripsi= %f ms\n", milliseconds);

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res2, devres2, (sizeof(ulint) * 2 *banyakdata), cudaMemcpyDeviceToHost);
	
	cudaFree(devc);
	cudaFree(devres2);
}

int *extendedEuclid (int a, int b){
	int *dxy = (int *)malloc(sizeof(int) *3);

	if (b ==0){
		dxy[0] =a; dxy[1] =1; dxy[2] =0;
		return dxy;
	}
	else{
		int t, t2;
		dxy = extendedEuclid(b, (a%b));
		t =dxy[1];
		t2 =dxy[2];
		dxy[1] =dxy[2];
		dxy[2] = t - a/b *t2;

		return dxy;
	}
}

void initenkripsi(ulint *m){
	srand(2018);
	for (int i = 0; i < banyakdata; i++) {
		m[i] = rand() % 256;
	}
}


int main(){
	ulint *m, p, q, pi, qi, n, *res, *res2;

	m = (ulint*)malloc(banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * sizeof(ulint));
	res2 = (ulint*)malloc(banyakdata * 2 * sizeof(ulint));

	p = 13;
	q = 23;
	n = 299;

	int *invers = extendedEuclid(p,q);
	pi = invers[1];
	qi = invers[2];

	initenkripsi(m);

	printf("<<<<<<<<<<<<<<Pesan Asli>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("m[%d] = %ld\n", i, m[i]);
	}

	printf("m[...]\n");
	printf("m[%d] = %ld\n", banyakdata-1, m[banyakdata-1]);

	enkripsiCUDA(m,n,res);

	printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("c[%d] = %ld\n", i, res[i]);
	}

	printf("c ...\n");
	printf("c[%d] = %ld\n", banyakdata-1, res[banyakdata-1]);

	dekripsiCUDA(res,p,q,pi,qi,n,res2);

	printf("<<<<<<<<<<<<<<Hasil Dekripsi>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("m[%d] = %ld 	m[%d] = %ld\n", 2*i, res2[2*i], 2*i+1, res2[2*i+1]);
	}

	printf("c ...\n");
	printf("c[%d] = %ld 	c[%d] = %ld\n", banyakdata * 2-2, res2[banyakdata * 2-2], banyakdata *2-1,res2[banyakdata*2-1]);


	free(m);
	free(res);
	free(res2);

	return 0;
}