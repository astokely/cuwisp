#include "cfrechet.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MAX3(x,y,z) (MAX(MAX(x,y),z))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MIN3(x,y,z) (MIN(MIN(x,y),z))


double l2norm(
		double *P, 
		double *Q, 
		int i, 
		int j, 
		int N
) {
	double px = P[i];
	double py = P[N + i];
	double pz = P[(2 * N) + i];
	double qx = Q[j];
	double qy = Q[N + j];
	double qz = Q[(2 * N) + j];
	double dpxqx = px - qx;
	double dpyqy = py - qy;
	double dpzqz = pz - qz;
	double delta = sqrt(
		pow(dpxqx, 2.0) 
		+ pow(dpyqy, 2.0) 
		+ pow(dpzqz, 2.0)
	);
	return delta;
}

double _frechet(
		double* P, 
		double* Q, 
		double* ca, 
		int i, 
		int j, 
		int N
) {
	int index = (i * N) + j;
	double dist;
	if (ca[index] > -1) {
		dist = ca[index];
		return dist;
	}
	else if (i == 0 && j == 0) {
		ca[index] = l2norm(
			P, Q, i, 
			j, N
		);
	}
	else if (i > 0 && j == 0) {
		double x = _frechet(
			P, Q, ca, 
			i - 1, 0, N
		);
		double y = l2norm(
			P, Q, i, 
			j, N
		);
		ca[index] = MAX(x, y);
	}
	else if (i == 0 && j > 0) {
		double x = _frechet(
			P, Q, ca, 
			0, j - 1, N
		);
		double y = l2norm(
			P, Q, i, 
			j, N
		);
		ca[index] = MAX(x, y);
	}	
	else if (i > 0 && j > 0) {
		double x = _frechet(
			P, Q, ca, 
			i-1, j, N
        );
		double y = _frechet(
			P, Q, ca, 
			i-1, j-1, N
		);
		double z = _frechet(
			P, Q, ca, 
			i, j-1, N
		);
		double m = MIN3(x, y, z);
		double norm = l2norm(
			P, Q, i, 
			j, N
		);
		ca[index] = MAX(m, norm);
	}
	else {
		ca[index] = INFINITY;
	}
	dist = ca[index];
	return dist;
}

double cFrechet(
		double* P, 
		int D1, 
		double* Q, 
		int D2, 
		double* ca, 
		int D3, 
		int i, 
		int j
) {
	int N = D1 / 3;
	i = N - 1;
	j = N - 1;
	double dist = _frechet(
		P, Q, ca, 
		i, j, N
	);
	return dist;
}


