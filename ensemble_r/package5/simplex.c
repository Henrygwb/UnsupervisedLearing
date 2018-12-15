/*==========================================================================================*/
/*                                                                                          */
/* Copyright (C) [2000]-[Oct 2017] Jia Li, Department of Statistics,                        */
/* The Pennsylvania State University, USA - All Rights Reserved                             */
/*                                                                                          */
/* Unauthorized copying of this file, via any medium is strictly prohibited                 */
/*                                                                                          */
/*                                                                                          */
/* NOTICE: All information contained herein is, and remains the property of The             */
/* Pennsylvania State University. The intellectual and technical concepts                   */
/* contained herein are proprietary to The Pennsylvania State University and may            */
/* be covered by U.S. and Foreign Patents, patents in process, and are protected            */
/* by trade secret or copyright law. Dissemination of this information or                   */
/* reproduction of this material is strictly forbidden unless prior written                 */
/* permission is obtained from Jia Li at The Pennsylvania State University. If              */
/* you obtained this code from other sources, please write to Jia Li.                       */
/*                                                                                          */
/*                                                                                          */
/*==========================================================================================*/
#include <math.h>
#include <stdio.h>
#include "matrix.h"
#define EPS 1.0e-3   /* the absolute precision, which should be adjusted 
			to the scale of the variables */
/* EPS can't be set too small because the equality condition used */
/* in computing mallows distance may have considerable precision error */

/*---------------------------------------------------------------------*/
/* Determines the maximum of those elements whose index is contained   */
/* in the supplied list ll, either with or without taking the absolute */
/* value, as flagged by iabf.                                          */
/*---------------------------------------------------------------------*/
void simp1(double **a, int mm, int *ll, int nll, int iabf, 
	   int *kp, double *bmax)
{
  int k;
  double test;
  
  if (nll <=0)
    *bmax = 0.0;
  else {
    *kp = ll[1];
    *bmax=a[mm+1][*kp+1];
    for (k=2; k<=nll; k++) {
      if (iabf == 0) 
	test = a[mm+1][ll[k]+1]-(*bmax);
      else
	test = fabs(a[mm+1][ll[k]+1])-fabs(*bmax);
      if (test > 0.0) {
	*bmax = a[mm+1][ll[k]+1];
	*kp = ll[k];
      }
    }
  }
}

/*--------------------------------------------------------*/
/* Locate a pivot element, taking degeneracy into account */
/*--------------------------------------------------------*/
void simp2(double **a, int m, int n, int *ip, int kp)
{
  int k,i;
  double qp, q0, q, q1;
  
  *ip = 0;
  for (i=1; i<=m; i++)
    if (a[i+1][kp+1]<-EPS) break; /* Any possible pivots? */

  if (i>m) return;
  q1 = -a[i+1][1]/a[i+1][kp+1];
  *ip = i;
  for (i=*ip+1; i<=m; i++) {
    if (a[i+1][kp+1] < -EPS) {
      q = -a[i+1][1]/a[i+1][kp+1];
      if (q<q1) {
	*ip = i;
	q1 = q;
      }
      else {
	if (q==q1) {
	  for (k=1; k<=n; k++) {
	    qp = -a[*ip+1][k+1]/a[*ip+1][kp+1];
	    q0 = -a[i+1][k+1]/a[i+1][kp+1];
	    if (q0 != qp) break;
	  }
	  if (q0<qp) *ip = i;
	}
      }
    }
  }
}

/*-------------------------------------------------------------------*/
/* Matrix operations to exchange a left-hand and right-hand variable */
/*-------------------------------------------------------------------*/
void simp3(double **a, int i1, int k1, int ip, int kp)
{
  int kk, ii;
  double piv;

  piv = 1.0/a[ip+1][kp+1];
  for (ii=1; ii<=i1+1; ii++)
    if (ii-1 != ip ) {
      a[ii][kp+1] *= piv;
      for (kk=1; kk<=k1+1; kk++)
	if (kk-1!=kp)
	  a[ii][kk] -= a[ip+1][kk]*a[ii][kp+1];
    }

  for (kk=1; kk<=k1+1; kk++)
    if (kk-1 != kp) a[ip+1][kk] *= -piv;
  a[ip+1][kp+1] = piv;
}

/*-----------------------------------------------------------------------*/
/* Simplex method for linear programming. Input paramenters a, m, n, mp, */
/* np, m1, m2, and m3, and output parameters a, icase, izrov, and iposv. */
/*-----------------------------------------------------------------------*/

void simplx(double **a, int m, int n, int m1, int m2, int m3, int *icase, 
	    int *izrov, int *iposv)
{
  int i, ip, is, k, kh, kp, nl1;
  int *l1, *l3;
  double q1, bmax;

  if (m != (m1+m2+m3)) {
    fprintf(stderr, "Bad input constraint counts in simplx\n");
    exit(1);
  }

  l1=(int *)calloc(n+2,sizeof(int));
  l3=(int *)calloc(m+1,sizeof(int));

  if (l1==NULL || l3==NULL) {
    fprintf(stderr, "Can't allocate space in simplx.\n");
    exit(1);
  }

  nl1 = n;
  for (k=1; k<=n; k++) l1[k] = izrov[k] = k;


  /* Initialize index list of columns admissible for exchange, and make
     all variables initially right-hand. */

  for (i=1; i<=m; i++) {
    if (a[i+1][1] < 0.0) {   /* constants b_i must be nonnegative */
      fprintf(stderr, "Bad input tableau in simplx a[%d+1][1]=%f\n",i,a[i+1][1]);
      exit(1);
    }
    iposv[i] = n+i;
  }

  if (m2+m3) {
    for (i=1; i<=m2; i++) l3[i] = 1;
    for (k=1; k<=n+1; k++) {
      q1 = 0.0;
      for (i=m1+1; i<=m; i++) q1 += a[i+1][k];
      a[m+2][k] = -q1;
    }
    for (; ;) {
      simp1(a, m+1, l1, nl1, 0, &kp, &bmax);
      if (bmax <= EPS && a[m+2][1] < -EPS) {
	*icase = -1;
	free(l3);
	free(l1);
	return;
      }
      else {
	if (bmax <=EPS && a[m+2][1]<=EPS) {
	  for (ip=m1+m2+1; ip <=m; ip++) {
	    if (iposv[ip] == (ip+n)) {
	      simp1(a, ip, l1, nl1, 1, &kp, &bmax);
	      if (bmax > EPS)
		goto one;
	    }
	  }
	  for (i=m1+1; i<=m1+m2; i++)
	    if (l3[i-m1] == 1)
	      for (k=1; k<=n+1; k++)
		a[i+1][k] = -a[i+1][k];
	  break;
	}
      }

      simp2(a, m, n, &ip, kp);
      if (ip==0) {
	*icase = -1;
	free(l1); free(l3);
	return;
      }
    one: simp3(a, m+1, n, ip, kp);
      if (iposv[ip] >= (n+m1+m2+1)) {
	for (k=1; k<=nl1; k++)
	  if (l1[k] == kp) break;
	--nl1;
	for (is=k; is <=nl1; is++) l1[is]=l1[is+1];
      }
      else {
	kh = iposv[ip]-m1-n;
	if (kh >= 1 && l3[kh]) {
	  l3[kh] = 0;
	  ++a[m+2][kp+1];
	  for (i=1; i<=m+2; i++)
	    a[i][kp+1]= -a[i][kp+1];
	}
      }
      is = izrov[kp];
      izrov[kp]=iposv[ip];
      iposv[ip] = is;
    }
  }
  
  for (; ; ) {
    simp1(a, 0, l1, nl1, 0, &kp, &bmax);
    if (bmax <= EPS) {
      *icase = 0;
      free (l1); free(l3);
      return;
    }
    simp2(a, m, n, &ip, kp);
    if (ip == 0) {
      *icase = 1;
      free(l1); free(l3);
      return;
    }
    simp3(a,m,n,ip,kp);
    is = izrov[kp];
    izrov[kp] = iposv[ip];
    iposv[ip] = is;
  }
}

