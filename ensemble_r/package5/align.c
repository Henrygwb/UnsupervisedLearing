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
#include "align.h"

int VERSION2=0;

//Sort in descending order
static int CompFcnFl(SORT_FLOAT *a, SORT_FLOAT *b)
{
  if (a->value > b->value)
    return (-1);
  if (a->value < b->value)
    return (1);
  return (0);
}

/*--------------------------------------------------------*/
/* Compute distance between all the pairs of clusters     */
/* based on membership of all the data samples.           */
/*--------------------------------------------------------*/

float dist2cls(int *cls1, int *cls2, int len, int id1, int id2)
{
  int i,j,k,m,n;
  float v1;
  
  for (i=0,v1=0.0; i<len;i++){
    if ((cls1[i]==id1 && cls2[i]!=id2)||(cls1[i]!=id1 && cls2[i]==id2))
      v1+=1.0;
  }
  return(v1);
}

float dist2cls_normalized(int *cls1, int *cls2, int len, int id1, int id2)
{
  int i,j,k,m,n;
  float v1,v2,v3,r1,r2;

  v1=v2=v3=0.0;
  for (i=0; i<len;i++){
    if (cls1[i]==id1 && cls2[i]!=id2)
      v1+=1.0; //in cluster 1 but not in cluster 2
    if (cls1[i]!=id1 && cls2[i]==id2)
      v2+=1.0; //in cluster 2 but not in cluster 1
    if (cls1[i]==id1 && cls2[i]==id2)
      v3+=1.0; //in both clusters
  }

  if (v1+v3==0.0) r1=1.0; else r1=v1/(v1+v3);//ratio of subtracted set
  if (v2+v3==0.0) r2=1.0; else r2=v2/(v2+v3);
  
  //return((r1+r2)/2.0);
  //Revised on Nov 7, 2017 to Jacaad index
  if (v3+v1+v2==0.0) r1=1.0; else r1=(v1+v2)/(v3+v1+v2);
  return(r1);
}

//Revised from dist2cls_normalized() above
//Compute Jacaard for two clusters represented by set of point ids
float Jacaard_pts(int *ptid1, int len1, int *ptid2, int len2)
{
  int i,j,k,m,n;
  float v1,v2,v3,r1,r2;
  int *cls1, *cls2;
  int id1, id2;
  int len;

  len=0;
  for (i=0;i<len1;i++) if (len<ptid1[i]) len=ptid1[i];
  for (i=0;i<len2;i++) if (len<ptid2[i]) len=ptid2[i];
  len++;
  
  cls1=(int *)calloc(len,sizeof(int));
  cls2=(int *)calloc(len,sizeof(int));
  for (i=0;i<len;i++) cls1[i]=cls2[i]=0;
  for (i=0;i<len1;i++) cls1[ptid1[i]]=1;
  for (i=0;i<len2;i++) cls2[ptid2[i]]=1;
  
  id1=1; id2=1;
  v1=v2=v3=0.0;
  for (i=0; i<len;i++){
    if (cls1[i]==id1 && cls2[i]!=id2)
      v1+=1.0; //in cluster 1 but not in cluster 2
    if (cls1[i]!=id1 && cls2[i]==id2)
      v2+=1.0; //in cluster 2 but not in cluster 1
    if (cls1[i]==id1 && cls2[i]==id2)
      v3+=1.0; //in both clusters
  }

  //Jacaard index
  if (v3+v1+v2==0.0) r1=1.0; else r1=(v1+v2)/(v3+v1+v2);

  free(cls1);
  free(cls2);

  return(r1);
  
}



void allpairs(int *cls1, int *cls2, int len, int n1, int n2, float *distmat)
//n1 is the #cluster in first clustering result, n2 is that for the second result
//dist[n1*n2], dist[0:n2-1] is the distance between cluster 0 in result 1 and cluster
//0, ..., n2-1 in result 2, so on so forth
{
  int i,j,k,m,n;

  if (!VERSION2) {
    for (i=0; i<n1; i++) {
      for (j=0; j<n2; j++)
	distmat[i*n2+j]=dist2cls(cls1,cls2,len,i,j);
    }
  } else {//Use the normalized distance with respect to each cluster's size
    for (i=0; i<n1; i++) {
      for (j=0; j<n2; j++) 
	distmat[i*n2+j]=dist2cls_normalized(cls1,cls2,len,i,j);
    }
  }
}

/*--------------------------------------------------------*/
/* IRM matching scheme.                                   */
/*--------------------------------------------------------*/
float match_fast(float *dist, float *p1_in, float *p2_in,  
		 int num1, int num2, float *wt)
     /* wt[num1*num2] stores the computed weights by optimization */
{
  int i,j,k,m,n, ii,jj;
  float minval;
  float *p1, *p2;
  int sum1, sum2; 
  float res;
  float TINY=1.0e-8;

  p1=(float *)calloc(num1,sizeof(float));
  p2=(float *)calloc(num2,sizeof(float));

  for (i=0; i<num1; i++) p1[i]=p1_in[i];
  for (i=0; i<num2; i++) p2[i]=p2_in[i];

  for (i=0;i<num1*num2;i++) wt[i]=0.0;

  sum1=sum2=0;

  while(sum1<num1 && sum2<num2) {
    minval=HUGE;
    ii=0, jj=0;
    for (i=0; i<num1; i++) {
      if (p1[i]<TINY) 
	continue;
      for (j=0; j<num2; j++) {
	if (p2[j]<TINY)
	  continue;
	if (dist[i*num2+j] < minval)
	  {
	    ii=i;
	    jj=j;
	    minval = dist[i*num2+j];
	  }
      }
    }
    
    if (p1[ii]<=p2[jj]) {
      wt[ii*num2+jj] = p1[ii];
      p2[jj] -= p1[ii];
      p1[ii]=0.0;
      sum1++;
      if (p2[jj]<TINY) sum2++;
    }
    else {
      wt[ii*num2+jj] = p2[jj];
      p1[ii] -= p2[jj];
      p2[jj]= 0.0;
      sum2++;
      if (p1[ii]<TINY) sum1++;
    }
  }  
  
  for (i=0,res=0.0;i<num1*num2;i++) res+=(wt[i]*dist[i]);

  free(p1);
  free(p2);
 
  return(res);

}


/*--------------------------------------------------------*/
/* Compute E(||X-Y||^p) by mallows distance matching.     */
/* The distance matrix ||X-Y||^p is assumed given by      */
/* *dist.                                                 */
/*--------------------------------------------------------*/
float match(float *dist, float *p1, float *p2, int num1, int num2, float *wt)
     /* wt[num1*num2] stores the computed weights by optimization */
{
  int i,j,k,m,n;
  int nconstraint, nvar;
  int icase;
  float res,v1,v2;
  double **a;
  int *iposv,*izrov;

  nvar = num1*num2;
  nconstraint = num1+num2;

  /* The unknowns are elements of wt[num1*num2] */

  /*-------------------------------------------*/
  /* Allocate space                            */
  /*-------------------------------------------*/

  m=(num1>num2)?num1:num2;
  a=(double **)calloc(m*2+3,sizeof(double *));
  for (i=0; i<m*2+3; i++) a[i]=(double *)calloc(m*m+2,sizeof(double));

  iposv=(int *)calloc(m*2+1,sizeof(int));
  izrov=(int *)calloc(m*m+1,sizeof(int));

  /*-------------------------------------------*/
  /* Use a to store all the constrains in the  */
  /* linear programming problem.               */
  /*-------------------------------------------*/

  for (i=0; i<nconstraint+3; i++)
    for (j=0; j<nvar+2; j++)
      a[i][j]=0.0;

  for (i=2, k=0; k<num1; i++, k++) {
    a[i][1] = p1[k];
    if (a[i][1]<0.0) a[i][1]=0.0;

    m = 2+(k+1)*num2;
    for (j=2+k*num2; j<m; j++) 
      a[i][j] = -1.0;
  }

  for (i=num1+2,k=0; k<num2; i++, k++)
    {
      a[i][1] = p2[k];
      if (a[i][1]<0.0) a[i][1]=0.0;

      for (j=2+k, n=0; n<num1; j+=num2, n++)
	a[i][j] = -1.0;
    }

  a[1][1] = 0.0;
  for (i=0; i<nvar; i++)
    a[1][i+2] = -dist[i];  // negative sign due to we want to minimize
                           // while simplx is for maximize

  simplx(a, nconstraint, nvar, 0, 0, nconstraint, &icase, izrov, iposv);
 
  if (icase!=0) {
    // icase=1, unbounded function, icase=-1, no feasible solution
    // use IRM distance
    fprintf(stderr, "Warning: Mallows distance replaced by IRM\n");
    res=match_fast(dist, p1, p2, num1, num2,wt);
  }
  else {
    res=-a[1][1];

    for (i=0;i<num1*num2;i++) wt[i]=0.0;
    for (j=0; j<nconstraint; j++) {
      if (iposv[j+1]-1<nvar)
	wt[iposv[j+1]-1]=a[j+2][1];
    }
  }


  for (i=0,v1=0.0;i<num1*num2;i++) v1+=wt[i]*dist[i];
  //fprintf(stderr, "res=%e, v1=%e\n",res,v1);

  m=(num1>num2)?num1:num2;
  for (i=0; i<m*2+3; i++) free(a[i]);
  free(a);
  free(iposv);
  free(izrov);

  return(res);
}

//Only for the purpose of showing intermediate information
void showinfo(float *dist, int n1, int n2, float *p1, float *p2)
{
  int i,j,k,m,n;
  
  fprintf(stdout, "======Distance matrix [%d x %d]============\n",n1,n2);
  for (i=0;i<n1;i++){
    for (j=0;j<n2;j++) {
      fprintf(stdout, "%e ",dist[i*n2+j]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "---------------------\n");
  fprintf(stdout, "--- Number of points in cluster result 1----\n");
  for (i=0;i<n1;i++)
    fprintf(stdout, "%e ",p1[i]);
  fprintf(stdout,"\n");
  fprintf(stdout, "--- Number of points in cluster result 2----\n");
  for (i=0;i<n2;i++)
    fprintf(stdout, "%e ",p2[i]);
  fprintf(stdout,"\n");
  fprintf(stdout, "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");  
}

/*--------------------------------------------------------*/
/* Compute Mallows distance for two bags of vectors.      */
/*--------------------------------------------------------*/
float alignclusters(int *cls1, int *cls2, int len, int n1, int n2, float *wt)
{
  int i,j,k,m,n;
  float *dist,mdist;
  float *p1, *p2, v1,v2;
  
  //Determine n1 or n2 automatically
  if (n1<=0) {
    n1=0;
    for (i=0;i<len;i++) {if (cls1[i]>n1) n1=cls1[i];}
    n1++;
  }
  if (n2<=0) {
    n2=0;
    for (i=0;i<len;i++) {if (cls2[i]>n2) n2=cls2[i];}
    n2++;
  }

  //compute all pairwise distances
  dist = (float *)calloc(n1*n2,sizeof(float));
  allpairs(cls1,cls2,len,n1,n2,dist);

  //compute weight for each cluster
  p1=(float *)calloc(n1,sizeof(float));
  p2=(float *)calloc(n2,sizeof(float));

  for (i=0,v1=v2=0.0;i<len;i++) {
    if (cls1[i]>=0) { p1[cls1[i]]+=1.0; v1+=1.0;}
    if (cls2[i]>=0) { p2[cls2[i]]+=1.0; v2+=1.0;}
  }

  //showinfo(dist,n1,n2,p1,p2);
  
  for (i=0;i<n1;i++) p1[i]/=v1;
  for (i=0;i<n2;i++) p2[i]/=v2;

  //Optimal transport matching (Wasserstein matching)
  mdist=match(dist, p1, p2, n1, n2,wt);

  //Forcefully set negative values in wt[] into 0.0
  for (i=0,m=n1*n2; i<m;i++){  if (wt[i]<0.0) wt[i]=0.0;}

  free(dist);
  free(p1); free(p2);

  return(mdist);
}

//The first len elements of cls[] are the reference clustering result
//ns is # bootstrap samples, including the reference clustering result
//len is the number of points in the data set
void align(int *cls, int ns, int len, float **wt, int **clsct_pt, float **dc_pt, int equalcls)
{
  int i,j,k,m,n;
  float *dist,mdist,*dc;
  int **clsarray, *clsct;

  if (ns<=1) {
    fprintf(stderr, "Wrong input: number of clustering results %d < 2\n",ns);
    exit(1);
  }

  clsarray=(int **)calloc(ns,sizeof(int *));
  for (i=0;i<ns;i++) clsarray[i]=cls+i*len;

  clsct=(int *)calloc(ns,sizeof(int));
  for (i=0;i<ns;i++){
    clsct[i]=0;
    for (j=0;j<len;j++) {if (clsarray[i][j]>clsct[i]) clsct[i]=clsarray[i][j];}
    clsct[i]++;
  }

  if (equalcls) {//force to have same number of clusters for each sample
    for (i=0,m=0;i<ns;i++){ if (m<clsct[i]) m=clsct[i];}
    if (clsct[0]<m) {
      fprintf(stderr, "Warning: the reference clustering has empty cluster\n");
    }
    for (i=0;i<ns;i++){ clsct[i]=m;}
  }

  dc=(float *)calloc(ns,sizeof(float));
  dc[0]=0.0; //distance to itself
  for (i=1,m=0;i<ns;i++) m+=clsct[i];
  m*=clsct[0];

  *wt=(float *)calloc(m,sizeof(float));
  for (i=1,m=0;i<ns;i++) {
    dc[i]=alignclusters(clsarray[i], clsarray[0], len, clsct[i], clsct[0], *wt+m);
    m+=clsct[i]*clsct[0];
  }

  *clsct_pt=clsct;
  *dc_pt=dc;
  free(clsarray);
  return;
}

void convertcode(int *cls, int len, int minsz)
{
  int i,j,k,m,n;
  int nc,*clsct,*code;

  for (i=0,nc=0;i<len;i++) {
    if (cls[i]>nc) nc=cls[i];
  }
  nc++;

  clsct=(int *)calloc(nc,sizeof(int));
  code=(int *)calloc(nc,sizeof(int));
  for (i=0;i<nc;i++) clsct[i]=0;
  for (i=0;i<len;i++) clsct[cls[i]]++;

  for (i=0,m=0;i<nc;i++) {
    if (clsct[i]<minsz) {code[i]=-1;}
    else {code[i]=m; m++;}
  }

  for (i=0;i<len;i++) cls[i]=code[cls[i]];
  
  free(clsct);
  free(code);
}

//wt[n1*n2], 1D array for matrix wt[n1][n2], each column corresponds
// to one cluster in the second result, and each row corresponds to
// one cluster in the first result
//reference cluster result is the seconde cluster result
//*code: 0 for match, 1 for split (a cluster in the second result is
//split into mulitple clusters in the first result), 2 for merge, 3 for others
//*nf: 1 for match, #clusters split into for split, #clusters needed to merge
//for merge
//code[n2], nf[n2] have been allocated with space
float maxentry(float *wt, int n1,int *id)
{
  int i,j,k;
  float v3;

  *id=0;
  v3=wt[0];
  for (j=1;j<n1;j++) {
    if (wt[j]>v3) {
      v3=wt[j];
      *id=j;
    }
  }
  return(v3);
}

//*nf is the number of clusters covered by the cluster in the reference
//return the value of the covered percentage of the cluster in the reference by
//all the clusters in the compared clustering that are covered by that cluster
//*maxv is the maxmum coverage by those clusters in the compared clustering that
//covered
float covercmp(float *wtcmp, float *wtref, int n1, int n2, int *nf, float *maxv,
	       int *maxid, float thred, float *coverage)
{
  int i,j,k,m,n;
  float v1,v2;

  //*maxv=v2 is the maximum of wtref[] of those clusters covered in the compared clustering
  //returned value v1 is the sum of such wtref[]'s. Hence v1>=v2.
  n=0;
  m=0;
  v1=v2=0.0;
  for (j=0;j<n1;j++) {
    if (wtcmp[j]>=thred) {
      if (coverage!=NULL) coverage[j]=wtref[j];
      n++;
      v1+=wtref[j];
      if (wtref[j]>v2) {
	v2=wtref[j];
	m=j;
      }
    } else {
      if (coverage!=NULL) coverage[j]=-1.0;
    }
  }
  
  *nf=n;
  *maxv=v2;
  *maxid=m;
  return(v1);
}

//Determine the matching, split, merge, and none-of-the-above status of the
//clusters.
//wt[n1*n2] is the matching weights
//Output is code[] and nf[]
void assess(float *wt, int n1, int n2, int *code, int *nf, float thred)
{
  int i,j,k,m,n,ii;
  float *wtcol, *wtrow; //column and row wise normalization of wt
  float *wtcmp, *wtref;
  float v1,v2,v3,v4,v5;
  int maxid;
  
  wtcol=(float *)calloc(n1*n2,sizeof(float));
  wtrow=(float *)calloc(n1*n2,sizeof(float));
  wtref=(float *)calloc(n1,sizeof(float));
  wtcmp=(float *)calloc(n1,sizeof(float));

  //row-wise normalization
  for (i=0;i<n1;i++) {
    for (j=0,v1=0.0;j<n2;j++)
      v1+=wt[i*n2+j];
    if (v1>0.0) {
      for (j=0;j<n2;j++)
	wtrow[i*n2+j]=wt[i*n2+j]/v1;
    }
    else {//treating empty clusters
      for (j=0;j<n2;j++)
	wtrow[i*n2+j]=0.0;
    }
  }

  //column-wise normalization
  for (i=0;i<n2;i++) {
    for (j=0,v2=0.0;j<n1;j++)
      v2+=wt[j*n2+i];
    if (v2>0.0) {
      for (j=0;j<n1;j++)
	wtcol[j*n2+i]=wt[j*n2+i]/v2;
    }
    else {//treating empty clusters
      for (j=0;j<n1;j++)
	wtcol[j*n2+i]=0.0;
    }
  }

  for (ii=0;ii<n2;ii++) {
    for (j=0;j<n1;j++) wtref[j]=wtcol[j*n2+ii];
    for (j=0;j<n1;j++) wtcmp[j]=wtrow[j*n2+ii];
    v1=covercmp(wtcmp, wtref, n1, n2, nf+ii, &v2,&maxid,thred,NULL);

    if (v2>=thred) {//match
      code[ii]=0;
    }
    else {
      if (v1>=thred) {//split
	code[ii]=1;
      } else {//possible merge or none of the three cases
	v3=maxentry(wtref,n1,&m);
	if (v3>=thred) {//possible merge
	  //m is the cluster which the reference clusters possibly merge to
	  v4=covercmp(wtcol+m*n2,wtrow+m*n2,n2,n1,nf+ii,&v5,&maxid, thred,NULL);

	  if (v4>=thred) {//merge
	    code[ii]=2;
	  }
	  else {
	    code[ii]=3;
	    nf[ii]=0;
	  }
	}
	else {
	  code[ii]=3;
	  nf[ii]=0;
	}
      }
    }
  }
  
  free(wtcol);
  free(wtrow);
  free(wtcmp);
  free(wtref);
  return;
}

//The matching,splitting, merging matrix res[n1*n2] cannot be of arbitrary pattern.
//It has to satisfy certain clear patterns. This is a parity check.
void paritycheck(float *res, int n1, int n2)
{
  int i,j,k,m,n;
  int m1,m2,m3;

  for (i=0;i<n2;i++) {
    m1=m2=m3=0;
    for (j=0;j<n1;j++) {
      if (res[j*n2+i]<0.0) m1++;
      if (res[j*n2+i]>=0.0 && res[j*n2+i]<=1.0) m2++;
      if (res[j*n2+i]>=2.0 && res[j*n2+i]<=3.0) m3++;
    }
    if (m1+m2+m3<n1) {
      fprintf(stderr, "Warning: m1+m2+m3<n1: m1=%d, m2=%d, m3=%d, n1=%d\n",m1,m2,m3,n1);
    }

    if (m3>1) {
      fprintf(stderr, "Warning: merge to more than 1: m1=%d, m2=%d, m3=%d, n1=%d\n",
	      m1,m2,m3,n1);
    }
    else {
      if (m3==1) {
	if (m3+m1<n1)
	  fprintf(stderr, "Warning: m3+m1<n1, m1=%d, m2=%d, m3=%d, n1=%d\n",m1,m2,m3,n1);
      }
      else {//m3==0 case
	if (m1+m2<n1)
	  fprintf(stderr, "Warning: m2+m1<n1, m1=%d, m2=%d, m3=%d, n1=%d\n",m1,m2,m3,n1);
      }
    }
  }
}

//Modified from assess2. Differ by having an extra output of the same dimension as
//wt[n1*n2], stores matching, splitting, and merging results res[n1*n2]
void assess2(float *wt, float *res, int n1, int n2, int *code, int *nf, float thred)
{
  int i,j,k,m,n,ii;
  float *wtcol, *wtrow; //column and row wise normalization of wt
  float *wtcmp, *wtref;
  float v1,v2,v3,v4,v5;
  int maxid;
  float *coverage;

  wtcol=(float *)calloc(n1*n2,sizeof(float));
  wtrow=(float *)calloc(n1*n2,sizeof(float));
  wtref=(float *)calloc(n1,sizeof(float));
  wtcmp=(float *)calloc(n1,sizeof(float));
  coverage=(float *)calloc(n1>n2 ? n1:n2, sizeof(float));

  //row-wise normalization
  for (i=0;i<n1;i++) {
    for (j=0,v1=0.0;j<n2;j++)
      v1+=wt[i*n2+j];
    if (v1>0.0) {
      for (j=0;j<n2;j++)
	wtrow[i*n2+j]=wt[i*n2+j]/v1;
    }
    else {//treating empty clusters
      for (j=0;j<n2;j++)
	wtrow[i*n2+j]=0.0;
    }
  }

  //column-wise normalization
  for (i=0;i<n2;i++) {
    for (j=0,v2=0.0;j<n1;j++)
      v2+=wt[j*n2+i];
    if (v2>0.0) {
      for (j=0;j<n1;j++)
	wtcol[j*n2+i]=wt[j*n2+i]/v2;
    }
    else {//treating empty clusters
      for (j=0;j<n1;j++)
	wtcol[j*n2+i]=0.0;
    }
  }

  for (ii=0;ii<n2;ii++) {
    for (j=0;j<n1;j++) wtref[j]=wtcol[j*n2+ii];
    for (j=0;j<n1;j++) wtcmp[j]=wtrow[j*n2+ii];
    v1=covercmp(wtcmp, wtref, n1, n2, nf+ii, &v2,&maxid,thred,coverage);
    for (j=0;j<n1;j++) { res[j*n2+ii]=coverage[j]; }	

    if (v2>=thred) {//match
      code[ii]=0;
    }
    else {
      if (v1>=thred) {//split
	code[ii]=1;
      } else {//possible merge or none of the three cases
	//It's possible that several components in reference merge to a component
	//in the bootstrap sample and one of merged members also matches with the
	//component in the bootstrap sample
	v3=maxentry(wtref,n1,&m);
	for (j=0;j<n1;j++) { res[j*n2+ii]=-1.0; }	
	if (v3>=thred) {//possible merge
	  //m is the cluster which the reference clusters possibly merge to
	  v4=covercmp(wtcol+m*n2,wtrow+m*n2,n2,n1,nf+ii,&v5,&maxid, thred,coverage);

	  if (v4>=thred) {//merge
	    code[ii]=2;
	    res[m*n2+ii]=2.0+coverage[ii];
	    if (coverage[ii]<0.0) { fprintf(stderr, "Paradox in assess2()\n");  }
	  }
	  else {
	    code[ii]=3;
	    nf[ii]=0;
	  }
	}
	else {
	  code[ii]=3;
	  nf[ii]=0;
	}
      }
    }
  }

  //Purely for testing purpose
  //Cases for a cluster in the reference: Pure match, impure match, split, merge, none-above
  paritycheck(res,n1,n2);
  
  free(wtcol);
  free(wtrow);
  free(wtcmp);
  free(wtref);
  free(coverage);
  return;
}

//Based on the matching weight matrix, generate the match,split,merge, etc.
//status for all the bootstrap samples
//Output: res, codect, nfave
//Basically, calls assess2() for all the bootstrap samples and summarize
//some information for each cluster in the reference, that is, codect, nfave
void MatchSplit(float *wt, float *res, int *numcls, int nbs,
		int **codect, float **nfave, float thred)
{
  int i,j,k,m,n, n0;
  int *code, *nf;

  n0=numcls[0];
  if (thred<=0.5){
    fprintf(stderr, "Warning: coverage threshold %f is too small\n",thred);
  }
  
  code=(int *)calloc(n0,sizeof(int));
  nf=(int *)calloc(n0,sizeof(int));

  for (j=0;j<n0;j++) {
    for (k=0;k<4;k++) {
      codect[j][k]=0;
      nfave[j][k]=0.0;
    }
  }
  
  //Matching, split, merge assessment for each bootstrap sample
  for (i=1,m=0;i<nbs;i++){
    assess2(wt+m*numcls[0],res+m*numcls[0],numcls[i],n0,code,nf,thred);
    m+=numcls[i];

    for (j=0;j<n0;j++) {
      codect[j][code[j]]++;
      nfave[j][code[j]]+=(float)nf[j];
    }
  }
  
  //Summarize the result for each cluster in the reference result
  for (j=0;j<n0;j++) {
    for (k=0;k<4;k++) {
      if (codect[j][k]>0)
	nfave[j][k]/=(float)codect[j][k];
    }
  }

  free(code);
  free(nf);
  return;
}

//Purely for testing purpose. Check whether the hard code has cases that shouldn't happen
//return 1 if there is the case of split (value=2), 0 otherwise
int checkcode(float *mycode, int n0, int n1)
{
  int i,j,k,m,n,m0,m1;
  
  for (i=0;i<n0;i++) {
    m0=m1=k=0;
    for (j=0;j<n1;j++) {
      if (mycode[j*n0+i]==1.0) m0++;
      if (mycode[j*n0+i]>=3.0 && mycode[j*n0+i]<=4.0) m1++;
      if (mycode[j*n0+i]==2.0) k++;
    }
    if (!((m0==1 && m1==0 && k==0)||(m0==0 && m1==0 && k>1)|| (m0==0 && m1==1 && k==0)|| m0+m1+k==0)){
      fprintf(stderr, "Warning!!: m0=%d, m1=%d, k=%d\n",m0,m1,k);
    } 
  }
  
  m1=0;
  for (j=0;j<n1;j++){
    m0=0;
    for (i=0;i<n0;i++){
      if (mycode[j*n0+i]>0.0) m0++;
      if (mycode[j*n0+i]==2.0) m1=1;
    }
    if (m0>1){ fprintf(stderr, "Warning: number of non-zero code in one row > 1: %d\n",m0); 
      fprintf(stderr, "input code to checkcode():\n");
      for (m=0;m<n1;m++) {
	for (n=0;n<n0;n++)
	  fprintf(stderr, "%.4f ",mycode[m*n0+n]);
	fprintf(stderr, "\n");
      }
    }  
  }
  
  return(m1);
}


//Based on match-split status stored in res[], do hard assign of components
//Code for assignment: 1: match, 2: reference cluster is split
//3+c: 0<=c<1: hard assigned, c is the column-wise normalized entry of weight matrix wt[]
//that is, the percentage of reference cluster covered by the cluster in the bootstrap sample
//Output: hdcode,npmatch,nimpmatch,nsplit,nmerge
//Return the #bootstrap samples leading to unassigned component in reference
int HardAssign(float *wt, float *res, float *hdcode, int *numcls, int nbs, int *cls, int len,
	       float thred)
{
  int i,j,k,m,n,ii,jj,mm;
  int n0,n1, m0,m1;
  float *myres,*mycode,*mywt;
  float v1,v2,v3;
  SORT_FLOAT *score;
  int *used0, *used1;
  int missed=0,num2=0;//#bootstrap samples leading to unassigned component in reference
  float *wtbt;
  int maxid;

  if (nbs<=1) return(0);
  
  n0=numcls[0];

  for (i=1,m=0;i<nbs;i++) m=(m<numcls[i])?numcls[i]:m;

  used0=(int *)calloc(n0,sizeof(int));
  used1=(int *)calloc(m,sizeof(int));

  wtbt=(float *)calloc(m,sizeof(float));
  
  score=(SORT_FLOAT *)calloc(n0,sizeof(SORT_FLOAT));
  for (i=0;i<n0;i++) {
    score[i].id=i;
    score[i].value=0.0;
    for (j=0;j<numcls[1];j++)
      score[i].value+=wt[j*n0+i];//prior of component i in reference
  }
  //score[j].id stores the original id of the jth largest value
  qsort((SORT_FLOAT *)score, n0, sizeof(SORT_FLOAT), CompFcnFl);

  for (ii=1,mm=0;ii<nbs;ii++){
    myres=res+mm*n0;
    mycode=hdcode+mm*n0;
    mywt=wt+mm*n0;
    mm+=numcls[ii];

    n1=numcls[ii];

    for (i=0;i<n0;i++) used0[i]=0;
    for (i=0;i<n1;i++) used1[i]=0;
    
    for (i=0;i<n0;i++) {
      m=0;
      for (j=0;j<n1;j++) {  mycode[j*n0+i]=0;}//reset
      for (j=0;j<n1;j++) {
	if (myres[j*n0+i]>=0.0 && myres[j*n0+i]<=1.0) {
	  mycode[j*n0+i]=1;
	  used0[i]=1;
	  used1[j]=1;
	  m++;
	}
      }

      if (m>1) {//split
	for (j=0;j<n1;j++) {
	  if (mycode[j*n0+i]) mycode[j*n0+i]=2;//correct to split from match
	}
      }
    }

    //empty cluster in the bootstrap sample will not be matched with anyone
    //force flag used1[] to 1
    for (j=0;j<n1;j++) wtbt[j]=0.0;
    for (k=0;k<len;k++) { wtbt[cls[ii*len+k]]+=1.0;}
    for (j=0,k=0;j<n1;j++) { if (wtbt[j]==0.0) {used1[j]=1; k=1;}}
    if (k) num2++;
    
    //If all consumed in either side, move to next sample
    for (i=0,m0=0;i<n0;i++) {if (!used0[i]) m0++;}
    for (i=0,m1=0;i<n1;i++) {if (!used1[i]) m1++;}
    if (m0==0||m1==0) {
      if (m0>0) missed++;
      continue;
    }

    //Deal with the left out components
    for (i=0;i<n0;i++) {
      m=score[i].id;//process in the order of largest component first
      v2=score[i].value;
      if (v2==0.0) {//empty cluster in the reference won't be assigned with anyone
	used0[m]=1;
	m0--;
	if (m0==0||m1==0) break;
      }
      
      if (!used0[m]) {//not matched or split yet
	//some components in the bootstrap sample haven't been assigned, otherwise
	//the loop would have been terminated.
	for (j=0;j<n1;j++) wtbt[j]=0.0;
	for (k=0;k<len;k++) {
	  if (cls[k]==m){
	    wtbt[cls[ii*len+k]]+=1.0;
	  }
	}
	//Find the maximum belonging component in reference
	for (j=0,k=0,v1=-1.0;j<n1;j++){
	  if (!used1[j] && wtbt[j]>v1) {
	    v1=wtbt[j];
	    k=j;
	  }
	}

	mycode[k*n0+m]=3+v1/(float)len;
	
	used0[m]=1;
	used1[k]=1;
	m0--;
	m1--;
	if (m0==0||m1==0) break;
      }
    }

    if (m0>0) missed++;
    //num2+=checkcode(mycode,n0,n1); //for debug only
  }// for (ii..)

  fprintf(stderr, "HardAssign:#bootstrap samples with less than %d components: %d\n",numcls[0],num2);
  free(score);
  free(used0);
  free(used1);
  free(wtbt);
  return(missed);
}


//Adjust the hard assignment code output by HardAssign()
//What it does: when a reference cluster is split, all the members in the bootstrap
//that are merged by the cluster in the reference will be checked to see whether each
//should be merged into some other cluster except for the largest member contained
//in the original cluster that splitted.
//A member whose largest cover according to the cluster label doesn't agree with the
//original split cluster will be moved to the cluster in the reference that provides
//the largest coverage.
//When the adjustment is done, both the hard code of the original split cluster in the
//reference and the new cluster in the reference for which that member was newly assigned to
//will be modifed.
//Modification: For the original split cluster: 2-->0 if the member is moved out, 2--> 1 if after
//moving out 1 member, there's only 1 member left and its coverage of the split cluster reaches
//the threshold, 2--> 3.c if after
//moving out 1 member, there's only 1 member left and its coverage of the split cluster is below
//the threshold. 2--> 2, if after removing one member, there are still 2 or more members.
//For the newly augmented cluster: if all the codes are 0 previously, then 0--> 3.c
//Otherwise, all the nonzero elements will be changed to 2.0 and the newly added
//member will have 0.0--> 2.0.
int AdjustHard(float *wt, float *hdcode, int *numcls, int nbs, int *cls,
	       int len, float thred)
{
  int i,j,k,m,n,ii,mm,jj,rid;
  int n0,n1, m0,m1,maxid,maxid0,num2=0;
  float *myres,*mycode,*mywt;
  float v1,v2,v3,v4;
  SORT_FLOAT *score;
  float *wtref, *wtbt;
  int adjusted=0;//#bootstrap samples with split that are adjusted to different results

  if (nbs<=1) return(0);
  n0=numcls[0];

  wtref=(float *)calloc(n0,sizeof(float));
  wtbt=(float *)calloc(n0,sizeof(float));

  //wtref[n0] is the marginal weight of the reference clusters
  for (i=0;i<n0;i++) {
    wtref[i]=0.0;
    for (j=0;j<numcls[1];j++)
      wtref[i]+=wt[j*n0+i];
    if (wtref[i]==0.0) {
      fprintf(stderr, "Warning: component %d is empty, containing no points in the cluster\n",i);
    }
  }

  for (ii=1,mm=0;ii<nbs;ii++){
    mycode=hdcode+mm*n0;
    mywt=wt+mm*n0;
    mm+=numcls[ii];

    n1=numcls[ii];
    
    for (rid=0;rid<n0;rid++){//only need to sweep once
      if (wtref[rid]<=0.0) continue; //no need to process empty cluster
      for (j=0,m=0,v1=0.0,maxid0=0;j<n1;j++){
	if (mycode[j*n0+rid]==2.0) {//split happens
	  m++;
	  if (v1<mywt[j*n0+rid]) {
	    v1=mywt[j*n0+rid]; //maximum weight
	    maxid0=j;
	  }
	}
      }

      if (m>0) {
	//Adjustment may occur
	for (j=0;j<n1;j++){// j is bootstrap sample cluster id
	  if (mycode[j*n0+rid]==2.0 && v1>mywt[j*n0+rid])
	    {//split happens and not the maximum merged member
	      for (jj=0;jj<n0;jj++) wtbt[jj]=0.0;
	      for (k=0;k<len;k++) {
		if (cls[ii*len+k]==j){
		  wtbt[cls[k]]+=1.0;
		}
	      }
	      //Find the maximum belonging component in reference
	      for (jj=0,maxid=0,v2=-1.0;jj<n0;jj++){
		if (wtbt[jj]>v2) {
		  v2=wtbt[jj];
		  maxid=jj;
		}
	      }
	      //No need to process an empty cluster
	      if (wtbt[rid]<v2) {//adjustment really happens
		//fprintf(stderr, "Adjust happens: bootstrap id=%d\n",ii);//comment out if desired
		//process the maxid component in reference
		for (jj=0,v3=0.0;jj<n1;jj++) {
		  if (mycode[jj*n0+maxid]>v3) v3=mycode[jj*n0+maxid];
		}

		if (v3==0.0) {
		  v4=mywt[j*n0+maxid]/wtref[maxid];
		  mycode[j*n0+maxid]=3.0+v4; 
		} else {
		  mycode[j*n0+maxid]=2.0; 
		  if (v3==1.0 || v3>=3.0) {
		    //if v3==2.0, no need to process more
		    for (jj=0;jj<n1;jj++) {
		      if (mycode[jj*n0+maxid]>=1.0) mycode[jj*n0+maxid]=2.0;
		    }
		  } 
		}

		//process the original split component
		mycode[j*n0+rid]=0.0;
		if (m==2) {//code update, 2--> 1
		  v4=mywt[maxid0*n0+rid]/wtref[rid];
		  if (v4>=thred) mycode[maxid0*n0+rid]=1.0;//pure match qualifies
		  else mycode[maxid0*n0+rid]=3.0+v4;
		}

		adjusted++;
	      }
	    }
	} //for (j...)
      } //if (m>0) ...    
    } // for (rid...
    //num2+=checkcode(mycode,n0,n1);//for debug
  } //for (ii...)

  free(wtref);
  free(wtbt);

  return(adjusted);
}

//Match summary per clustering result
void MatchSum(float *res, int n1, int n2, int *npmatch, int *nimpmatch, int *nsplit, int *nmerge, float thred)
{
  int i,j,k,m,n;
  int m1,m2,m3;
  float v1;

  *npmatch=0;
  *nimpmatch=0;
  *nsplit=0;
  *nmerge=0;

  for (i=0;i<n2;i++) {
    m1=m2=m3=0;
    v1=-1.0;
    for (j=0;j<n1;j++) {
      if (res[j*n2+i]<0.0) m1++;
      if (res[j*n2+i]>=0.0 && res[j*n2+i]<=1.0) {
	m2++;
	if (v1<res[j*n2+i]) v1=res[j*n2+i];
      }
      if (res[j*n2+i]>=2.0 && res[j*n2+i]<=3.0) m3++;
    }
    if (m3>0) {
      *nmerge=*nmerge+1;
    } else {
      if (m2==1) {
	*npmatch=*npmatch+1;
      }
      else {
	if (m2>1) {
	  if (v1>=thred) {
	    *nimpmatch=*nimpmatch+1;
	  }
	  else { *nsplit=*nsplit+1;}
	}
      }
    }
  }
}

//For the purpose of summarizing per bootstrap sample information
void MatchInfo(float *res, int *numcls, int nbs, float thred, FILE *outfile)
{
  int i,j,k,m,n,n0;
  int *npmatch, *nimpmatch, *nsplit, *nmerge;
  int m1,m2,m3,m4;

  npmatch=(int *)calloc(nbs,sizeof(int));
  nimpmatch=(int *)calloc(nbs,sizeof(int));
  nsplit=(int *)calloc(nbs,sizeof(int));
  nmerge=(int *)calloc(nbs,sizeof(int));

  n0=numcls[0];
  for (i=1,m=0;i<nbs;i++){
    MatchSum(res+m*numcls[0],numcls[i],n0,npmatch+i,nimpmatch+i,nsplit+i,nmerge+i,thred);
    m+=numcls[i];
  }
  
  m1=m2=m3=0;
  for (i=1;i<nbs;i++){
    if (npmatch[i]==n0) m1++;
    if (npmatch[i]+nimpmatch[i]==n0) m2++;
    if (npmatch[i]+nimpmatch[i]+nsplit[i]==n0) m3++;
  }

  fprintf(outfile, "Summary of bootstrap sample match\n Threshold for sufficient coverage %f\n",thred);
  fprintf(outfile, "pure match=%d, up to impure match=%d, up to split=%d\n",m1,m2,m3);

  m1=m2=m3=m4=0;
  for (i=1;i<nbs;i++){
    m1+=npmatch[i];
    m2+=nimpmatch[i];
    m3+=nsplit[i];
    m4+=nmerge[i];
  }
  fprintf(outfile, "Average #clusters: pure match=%f, impure match=%f, split=%f, merge=%f\n",
	  (float)m1/(nbs-1.0), (float)m2/(nbs-1.0), (float)m3/(nbs-1.0),(float)m4/(nbs-1.0));
  
  free(npmatch);
  free(nimpmatch);
  free(nsplit);
  free(nmerge);
  
}

//-------- For computing confidence set for all the matched/split clusters for --
//-------- each cluster in the reference clustering with #clusters=n2
//??? Insert confset2.c here
static int CompFcn(SORT_INT *a, SORT_INT *b)
{
  if (a->value > b->value)
    return (1);
  if (a->value < b->value)
    return (-1);
  return (0);
}

void SortInt(int *org, int *buf, int *invid, int sz)
{
  int i,j,k;
  SORT_INT *score;

  score=(SORT_INT *)calloc(sz,sizeof(SORT_INT));
  if (score==NULL) {
    fprintf(stderr, "Unable to allocate space in SortInt.\n");
    exit(0);
  }
  for (j=0;j<sz;j++) {
    score[j].id=j;
    score[j].value=org[j];
  }
  qsort((SORT_INT *)score, sz, sizeof(SORT_INT), CompFcn);

  for (j=0;j<sz;j++) {
    buf[j]=org[score[j].id];
    invid[j]=score[j].id; //store the original id for the new order               
  }

  free(score);
}

//cls[0] has space allocated
void NewCLink(CLink *cls, int np)
{
  cls->n=np;
  cls->id=(int *)calloc(np,sizeof(int));
}

void FreeCLink(CLink *cls)
{
  if (cls->n >0) free(cls->id);
}

void Readcls(FILE *infile, int *ns, CLink **clist)
{
  int i,j,k,m,n;

  *ns=0;
  while (!feof(infile)) {
    fscanf(infile, "%d ", &m);
    for (j=0;j<m;j++) fscanf(infile, "%d", &n);
    fscanf(infile, "\n");
    *ns= *ns+1;
  }

  rewind(infile);

  *clist=(CLink *)calloc(*ns,sizeof(CLink));
  
  for (i=0;i<*ns;i++) {
    fscanf(infile, "%d ", &m);
    NewCLink((*clist)+i,m);
    for (j=0;j<m;j++) {
      fscanf(infile, "%d", &n);
      (*clist+i)->id[j]=n;
    }
    fscanf(infile, "\n");
  }
}

//Sort the ids in each clist[] so that they are in ascending order
void Sortcls(CLink *clist, int ns)
{
  int i,j,k,m;
  int *buf, *invid;

  m=0;
  for (i=0;i<ns;i++) {
    if (m<clist[i].n) m=clist[i].n;
  }
  
  buf=(int *)calloc(m,sizeof(int));
  invid=(int *)calloc(m,sizeof(int));
  for (i=0;i<ns;i++) {
    SortInt(clist[i].id, buf, invid, clist[i].n);
    for (j=0;j<clist[i].n;j++)
      clist[i].id[j]=buf[j];
  }
  free(buf);
}

//Generate a list of numbers from 0 to *nids-1 for the original IDs
//in clist[].
void MapIds(CLink *clist, int ns, int *maxid, int *nids, int **id2num, int **num2id)
{
  int i,j,k,m,n;

  m=0;
  for (i=0;i<ns;i++)
    for (j=0;j<clist[i].n;j++)
      if (clist[i].id[j]>m) 
	m=clist[i].id[j];
  m++;
  *maxid=m; //maximum value of id plus 1

  *id2num=(int *)calloc(m,sizeof(int)); 
  for (i=0;i<m;i++) (*id2num)[i]=0;

  for (i=0;i<ns;i++)
    for (j=0;j<clist[i].n;j++)
      (*id2num)[clist[i].id[j]]++;

  k=0;
  for (i=0;i<m;i++) {
    if ((*id2num)[i]==0) (*id2num)[i]=-1; //non-existing ids
    else {
      (*id2num)[i]=k; 
      k++;
    }
  }
  
  *nids=k;
  *num2id=(int *)calloc(k,sizeof(int));
  for (i=0;i<m;i++)
    if ((*id2num)[i]>=0) {
      (*num2id)[(*id2num)[i]]=i;
    }
}

//For a given point with id, compute the number of clusters that
//contain this point. Whether a cluster should be considered is indicated
//by keepcls[ns].
int ClusterInclude(CLink *clist, int ns, unsigned char *keepcls, int id, unsigned char *touched)
{
  int i,j,k,m,n;

  n=0;
  for (i=0;i<ns;i++) {
    touched[i]=0;
    if (keepcls[i]==0) continue;
    for (j=0;j<clist[i].n;j++) {
      if (clist[i].id[j]==id) {
	n++;
	touched[i]=1;
  	break;
      }
      else {
	if (clist[i].id[j]>id) break; //We can do this assuming the ids are sorted
      }
    }
  }

  return(n);
}

//Find the confidence set. The only output is pts[nids], keepcls[ns]
//The space for pts[] and keepcls[] has been allocated.
//alpha is the percentage of clusters that can be excluded at most.
void ConfidenceSet(CLink *clist, int ns, int nids, int *id2num, int *num2id,
		   unsigned char *pts, unsigned char *keepcls, float alpha)
{
  int i,j,k,m,n,mm,nn;
  int nexclude, ncv;
  unsigned char *touched, *buf;
  int *p, mv;

  p=(int *)calloc(nids,sizeof(int));
  touched =(unsigned char *)calloc(ns,sizeof(char));
  buf =(unsigned char *)calloc(ns,sizeof(char));

  nexclude=(int)(alpha*(float)ns);
  ncv=ns;

  for (i=0;i<nids;i++) pts[i]=1; //indicator for the inclusion of each point
  for (i=0;i<ns;i++) keepcls[i]=1; //indicator for each cluster being covered

  while (ncv>ns-nexclude) {
    //For each existing point and existing cluster, compute p[]
    for (i=0;i<nids;i++) p[i]=0;
    m=-1; mv=ns+1;
    for (i=0;i<nids;i++) {
      if (pts[i]==0) continue; //already excluded
      p[i]=ClusterInclude(clist, ns, keepcls, num2id[i], buf);
      if (p[i]<mv) {
	mv=p[i];
	m=i;
	for (j=0;j<ns;j++) touched[j]=buf[j];
      }
    }

    if (ncv-mv>=ns-nexclude) {
      //exclude points and clusters
      for (j=0;j<ns;j++) 
	if (touched[j]) {
	  keepcls[j]=0;
	}
      ncv-=mv;

      for (i=0;i<nids;i++) pts[i]=0; //indicator for the inclusion of each point
      for (i=0;i<ns;i++) {
	if (keepcls[i]==0) continue;
	for (j=0;j<clist[i].n;j++) pts[id2num[clist[i].id[j]]]=1;
      }
    } else { //terminate the loop
      break;
    }
  }

  free(p);
  free(buf);
  free(touched);
}


//  int nbs,len
//  float alpha=0.1, v1,v2,v3;
//  CLink *clist;
void confset(CLink *clist, int nbs, float alpha, int **confpts, int *npts, unsigned char **keepcls_pt, float **cvp_pt, float *tightness, float *coverage)
{
  int i,j,k,m,n,i1,i2;
  int minsz=0;
  int *numcls;
  float v1,v2,v3;
  
  /*----------------------------------------------------------------*/
  /*----------------- Read in data ---------------------------------*/
  /*----------------------------------------------------------------*/

  Sortcls(clist,nbs);

  /*------------------------------------------------------------*/
  /*- Find Confidence Set                                       */
  /*------------------------------------------------------------*/
  
  int nids, maxid, *id2num, *num2id;
  unsigned char *pts, *keepcls;
  float *cvp;
  
  MapIds(clist, nbs, &maxid, &nids, &id2num, &num2id);

  pts=(unsigned char *)calloc(nids, sizeof(char));
  keepcls=(unsigned char *)calloc(nbs, sizeof(char));
  ConfidenceSet(clist, nbs, nids, id2num, num2id, pts, keepcls, alpha);

  /*------------------------------------------------------------*/
  /*- Output the result                                         */
  /*------------------------------------------------------------*/
  for (i=0,m=0;i<nids;i++) if (pts[i]) m++;
  *npts=m;
  *confpts=(int *)calloc(m,sizeof(int));
  for (i=0,k=0;i<nids;i++)
    if (pts[i]) {
      (*confpts)[k]=num2id[i];
      k++;
    }

  //For included clusters, compute the percentage w.r.t. the confidence set
  //For excluded clusters, compute the percentage of points in each cluster
  //that are included in the confidence set
  cvp=(float *)calloc(nbs,sizeof(float));
  i1=i2=0;
  v1=v2=0.0;
  for (i=0;i<nbs;i++) {
    if (keepcls[i]) {
      cvp[i]=((float)clist[i].n)/((float)m);
      v1+=cvp[i];
      i1++;
    }
    else {
      for (j=0,k=0;j<clist[i].n;j++)
	if (pts[id2num[clist[i].id[j]]]) k++;
      cvp[i]=((float)k)/((float)clist[i].n); //percentage of covered points
      v2+=cvp[i];
      i2++;
    }
  }

  if (i1>0) v1/=(float)i1;
  if (i2>0) v2/=(float)i2;

  *tightness=v1;
  *coverage=v2;

  *keepcls_pt=keepcls;
  *cvp_pt=cvp;
  free(pts);
  free(id2num);
  free(num2id);
}


//============================================
void MatchCluster(float *res, int *numcls, int nbs, float thred, int *cls, int len, CLink ***clist2, int **nsamples, int usesplit)
{
  int i,j,k,m,n,n0,n1,n2,ii,k1,k2,jj,kk;
  int *matched, nmatch;
  int m1,m2,m3,m4;
  float *res_cur;
  float v1;
  
  *nsamples=(int *)calloc(numcls[0],sizeof(int));
  *clist2=(CLink **)calloc(numcls[0], sizeof(CLink *));
  for (ii=1,n0=0;ii<nbs;ii++){  if (n0<numcls[ii]) n0=numcls[ii];}
  matched=(int *)calloc(n0,sizeof(int));

  n2=numcls[0];
  for (i=0;i<n2;i++) {//process each cluster in the reference
    for (ii=1,m=0,nmatch=0;ii<nbs;ii++){
      n1=numcls[ii];
      res_cur=res+m*n2;
      
      m1=m2=m3=0;
      v1=-1.0;
      for (j=0;j<n1;j++) {
	if (res_cur[j*n2+i]<0.0) m1++;
	if (res_cur[j*n2+i]>=0.0 && res_cur[j*n2+i]<=1.0) {
	  m2++;
	  if (v1<res_cur[j*n2+i]) v1=res_cur[j*n2+i];
	}
      }
      
      if (m2>=1){//match or split
	if (m2==1 || v1>=thred) {//match
	  nmatch++;
	}
	else { //split
	  if (usesplit) nmatch++;
	}
      }
      m+=numcls[ii];  
    }

    //Reference cluster itself is always recorded
    (*clist2)[i]=(CLink *)calloc(nmatch+1,sizeof(CLink));
    (*nsamples)[i]=nmatch+1; //nsamples[i] is the number of samples used for confset for cluster i

    //Record the point ids for the reference cluster
    for (j=0,k=0;j<len;j++) {
      if (cls[j]==i){ k++; }
    }
    
    NewCLink((*clist2)[i], k);
    
    for (j=0,k=0;j<len;j++) {
      if (cls[j]==i){
	(*clist2)[i][0].id[k]=j; //point id
	k++;
      }
    }
    
    //Record point ids for the matched clusters in the other partitions
    for (ii=1,m=0,kk=0;ii<nbs;ii++){
      n1=numcls[ii];
      res_cur=res+m*n2;
      for (j=0;j<n1;j++) matched[j]=0;
      
      m1=m2=m3=0;
      v1=-1.0; k2=0;
      for (j=0;j<n1;j++) {
	if (res_cur[j*n2+i]<0.0) m1++;
	if (res_cur[j*n2+i]>=0.0 && res_cur[j*n2+i]<=1.0) {
	  m2++;
	  matched[j]=1;
	  if (v1<res_cur[j*n2+i]) { v1=res_cur[j*n2+i]; k2=j;}
	}
      }
      
      if (usesplit!=1 && m2>1) {
	for (j=0;j<n1;j++) matched[j]=0;
	if (v1>=thred) //reset matched for impure match case
	  matched[k2]=1;//The single 1 in matched[]
      }
     
      //List point ids for the matched cluster(s)
      if (m2>=1 && (m2==1 || v1>=thred || usesplit)) {
	for (j=ii*len,k=0;j<ii*len+len;j++) {
	  if (matched[cls[j]]){
	    k++;
	  }
	}
	
	NewCLink((*clist2)[i]+kk+1, k);

	for (j=ii*len,k=0;j<ii*len+len;j++) {
	  if (matched[cls[j]]){
	    (*clist2)[i][kk+1].id[k]=j-ii*len; //point id
	    k++;
	  }
	}

	kk++; //index for nmatch
      }

      m+=numcls[ii];  
    }
  }

  free(matched);
  
}

//confpts[numcls], npts[numcls], avetight[numcls], avecov[numcls], avejacaard[numcls], rinclude[numcls] have space allocated
//confpts[numcls] is an empty link to be allocated 
//confpts[i] is a list of point ids that are in the confidence set
//The size of confpts[i] is output npts[i]
//avetight[i] and avecov[i] are outputs: average tightness and average coverage
//for the confidence set for the ith reference cluster
//avejacaard[i] is the average jacaard index between the reference
//and the other mathced clusters.
//rinclude[i] is the percentage of clusters fully covered by the confidence set
//csetdist[numcls*numcls] is a square matrix recording the pairwise Jacaard distance
//between the confidence set of two reference clusters, space allocated assumed
//Only output: confpts, npts, avetight, avecov, avejacaard, rinclude, csetdist
void AveConfset(CLink **clist2, int numcls, int *nsamples, float alpha, int **confpts, int *npts, float *avetight, float *avecov, float *avejacaard, float *rinclude, float *csetdist)
{
  int i,j,k,m,n;
  unsigned char *keepcls;
  float *cvp, v1;
  CLink cset1, cset2;

  for (i=0;i<numcls;i++) {
    if (nsamples[i]>1) {
      confset(clist2[i], nsamples[i], alpha, confpts+i, npts+i, &keepcls, &cvp, avetight+i, avecov+i);
      for (j=0, rinclude[i]=0.0;j<nsamples[i];j++)
	if (keepcls[j]) rinclude[i]+=1.0;
      rinclude[i]/=(float)nsamples[i];

      v1=0.0;
      for (j=1;j<nsamples[i];j++)
	v1+=Jacaard_pts(clist2[i][0].id, clist2[i][0].n, clist2[i][j].id, clist2[i][j].n);
      avejacaard[i]=v1/(float)(nsamples[i]-1);

      free(keepcls);
      free(cvp);
    }
    else {
      //artificial setup, confidence set is the reference set
      //which is not matched with any
      avetight[i]=avecov[i]=1.0; 
      rinclude[i]=1.0;
    }
  }

  for (i=0;i<numcls;i++) {
    csetdist[i*numcls+i]=0.0;
    for (j=i+1;j<numcls;j++){
      csetdist[i*numcls+j]=Jacaard_pts(confpts[i], npts[i], confpts[j],npts[j]);
      csetdist[j*numcls+i]=csetdist[i*numcls+j];
    }
  }

}
