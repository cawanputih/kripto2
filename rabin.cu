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

ulint modexp(ulint a, ulint b, ulint c) {
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

void enkripsi(ulint m, ulint n, ulint *res) {
	*res = m*m % n;
}

void dekripsi(ulint c, ulint p, ulint q, ulint pi, ulint qi, ulint n, ulint *res) {
	ulint mp = modexp(c, (p+1)/4, p);
	ulint mq = modexp(c, (q+1)/4, q);

	*res = (pi * p * mq + qi * q * mp) % n;
	*(res+1) = (pi * p * mq - qi * q * mp) % n;
}

void kernelenk(ulint *m, ulint n, ulint *res) {
	for (int i = 0; i < banyakdata; i++)
	{
		enkripsi(m[i], n, res + i);
	}
}

void kerneldek(ulint *c, ulint p, ulint q, ulint pi, ulint qi, ulint n, ulint *res) {
	for (int i = 0; i < banyakdata; i++)
	{
		dekripsi(c[i], p, q, pi, qi, n, res + 2*i);
	}
}

void enkripsiCUDA(ulint *m, ulint n, ulint *res) {
	clock_t begin = clock();
		kernelenk(m,n,res);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent/1000);
}

void dekripsiCUDA(ulint *c, ulint p, ulint q, ulint pi, ulint qi, ulint n, ulint *res2) {
	clock_t begin = clock();
		kerneldek(c,p,q,pi,qi,n,res2);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent/1000);
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