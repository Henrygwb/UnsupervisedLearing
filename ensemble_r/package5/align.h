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
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include "matrix.h"

extern int VERSION2;

typedef struct 
{
  int id;
  float value;
} SORT_FLOAT;

typedef struct 
{
  int n;
  int *id;
} CLink;

typedef struct 
{
  int id;
  int value;
} SORT_INT;


//simplex.c
extern void simplx(double **, int, int, int, int, int, int *, int *, int *);

//align.c
extern float dist2cls(int *cls1, int *cls2, int len, int id1, int id2);
extern void allpairs(int *cls1, int *cls2, int len, int n1, int n2, float *distmat);
extern float match_fast(float *dist, float *p1_in, float *p2_in,  
			int num1, int num2, float *wt);
extern float match(float *dist, float *p1, float *p2, int num1, int num2, float *wt);
extern float alignclusters(int *cls1, int *cls2, int len, int n1, int n2, float *wt);
extern void align(int *cls, int ns, int len, float **wt, int **clsct_pt, float **dc_pt,int);
extern void convertcode(int *cls, int len, int minsz);
extern void assess(float *wt, int n1, int n2, int *code, int *nf, float thred);
extern void assess2(float *wt, float *res, int n1, int n2, int *code, int *nf, float thred);
extern void MatchSum(float *res, int n1, int n2, int *npmatch, int *nimpmatch,
		     int *nsplit, int *nmerge, float thred);
extern void MatchSplit(float *wt, float *res, int *numcls,
		       int nbs, int **codect,float **nfave, float thred);
extern void MatchInfo(float *res, int *numcls, int nbs, float thred, FILE *outfile);
extern int HardAssign(float *wt, float *res, float *hdcode, int *numcls, int nbs, int *cls, int len,
		      float thred);
extern int AdjustHard(float *wt, float *hdcode, int *numcls, int nbs, int *cls,
		      int len, float thred);

//---- From confset.c main program, now subroutines in align.c
extern void confset(CLink *clist, int nbs, float alpha, int **confpts, int *npts, unsigned char **keepcls_pt, float **cvp_pt, float *tightness, float *coverage);
extern void MatchCluster(float *res, int *numcls, int nbs, float thred, int *cls, int len, CLink ***clist2, int **nsamples, int usesplit);
extern void AveConfset(CLink **clist2, int numcls, int *nsamples, float alpha, int **confpts, int *npts, float *avetight, float *avecov, float *avejacaard, float *rinclude, float *csetdist);



