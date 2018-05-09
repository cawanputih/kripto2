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

void modexp(ulint a, ulint b, ulint c, ulint* res) {
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

void kernelenk(ulint *m, ulint e, ulint n, ulint *res) {
	for (int i = 0; i < banyakdata; i++)
	{
		modexp(m[i], e, n, res + i);
	}
}

void kerneldek(ulint *c, ulint d, ulint n, ulint *res) {
	for (int i = 0; i < banyakdata; i++)
	{
		modexp(c[i], d, n, res + i);
	}
}

void enkripsiCUDA(ulint *m, ulint e, ulint n, ulint *res) {
	clock_t begin = clock();
		kernelenk(m,e,n,res);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent/1000);
	printf("\n<<<<<<<<<<<<<<HASIL KE CPU>>>>>>>>>>>>>>>\n");
}

void dekripsiCUDA(ulint *c, ulint d, ulint n, ulint *res2) {
	clock_t begin = clock();
		kerneldek(c,d,n,res2);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent/1000);
	printf("\n<<<<<<<<<<<<<<HASIL KE CPU>>>>>>>>>>>>>>>\n");
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