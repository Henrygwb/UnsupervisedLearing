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
#include <string.h>

int main(argc, argv)
     int argc;
     char *argv[];
{
  char infilename[100], outfilename[100], parfilename[100];
  char wtfilename[100], cfilename[100],hfilename[100];
  FILE *infile, *outfile, *parfile, *wtfile, *cfile, *hfile;
  int i,j,k,m,n;
  int nbs,len,minsz=0;
  int *cls;
  float *wt, *res, *hdcode, *dist;
  int *numcls;
  float thred=0.8;
  int equalcls=0;
  int usesplit=0;
  float alpha=0.1;
  
  /*----------------------------------------------------------------*/
  /*---------------- Read in parameters from command line-----------*/
  /*----------------------------------------------------------------*/

  i = 1;
  while (i <argc)
    {
      if (*(argv[i]) != '-')
        {
          printf("**ERROR** bad arguments\n");
          exit(1);
        }
      else
        {
          switch(*(argv[i++] + 1))
            {
            case 'i':
              strcpy(infilename,argv[i]);
              break;
            case 'o':
              strcpy(outfilename,argv[i]);
              break;
            case 'p':
              strcpy(parfilename,argv[i]);
              break;
            case 'w':
              strcpy(wtfilename,argv[i]);
              break;
            case 'h':
              strcpy(hfilename,argv[i]);
              break;
            case 'c':
              strcpy(cfilename,argv[i]);
              break;
	    case 'b':
              sscanf(argv[i],"%d",&nbs);
              break;
            case 'l':
              sscanf(argv[i],"%d",&len);
              break;
            case 't':
              sscanf(argv[i],"%f",&thred);
              break;
            case 'a':
              sscanf(argv[i],"%f",&alpha);
              break;
	    case 'e':
	      equalcls=1;
	      i--;
	      break;
	    case 's':
	      usesplit=1;
	      i--;
	      break;
            case '2':
              VERSION2=1;
	      i--;
              break;
            default:
              {
                printf("**ERROR** bad arguments\n");
                exit(1);
              }
            }
          i++;
        }
    }

  /*----------------------------------------------------------------*/
  /*--------------------- open files -------------------------------*/
  /*----------------------------------------------------------------*/
  
  infile = fopen(infilename, "r");
  if (infile == NULL)
    {
      printf("Couldn't open input data file \n");
      exit(1);
   }

  outfile = fopen(outfilename, "w");
  if (outfile == NULL)
    {
      printf("Couldn't open output file \n");
      exit(1);
    }
  
  hfile = fopen(hfilename, "w");
  if (hfile == NULL)
    {
      printf("Couldn't open output file \n");
      exit(1);
    }
  
  parfile = fopen(parfilename, "w");
  if (parfile == NULL)
    {
      printf("Couldn't open parameter file \n");
      exit(1);
    }
  
  wtfile = fopen(wtfilename, "w");
  if (wtfile == NULL)
    {
      printf("Couldn't open weight file \n");
      exit(1);
    }
  
  cfile = fopen(cfilename, "w");
  if (cfile == NULL)
    {
      printf("Couldn't open weight file \n");
      exit(1);
    }
  

  /*----------------------------------------------------------------*/
  /*----------------- Read in data ---------------------------------*/
  /*----------------------------------------------------------------*/

  m=0;
  while (!feof(infile)){
    fscanf(infile, "%d\n",&n);
    m++;
  }
  if (m%nbs!=0) {
    fprintf(stderr, "Wrong input: #lines=%d mod #samples=%d!=0\n",m,nbs);
    exit(0);
  } else {
    len=m/nbs;
  }
  
  rewind(infile);
  cls=(int *)calloc(len*nbs,sizeof(int));

  for (m=0,n=len*nbs;m<n;m++) {
    if (feof(infile)) {
      fprintf(stderr, "Error: not enough data in input file\n");
      exit(0);
    }
    fscanf(infile, "%d\n", cls+m);
  }

  /*------------------------------------------------------------*/
  /*- Alignment done here to get the key matching matrix wt[]  -*/
  /*------------------------------------------------------------*/
  
  align(cls,nbs,len,&wt,&numcls,&dist,equalcls);

  fprintf(stderr, "nbs=%d, len=%d\n",nbs,len);

  /*------------------------------------------------------------*/
  /*- Output the weight matrix and the distances between        */
  /*- clustering results.                                       */
  /*------------------------------------------------------------*/
  for (i=0;i<nbs;i++){
    fprintf(parfile, "%d %f\n", numcls[i],dist[i]);
  }

  for (i=1,m=0;i<nbs;i++){
    for (j=0;j<numcls[i];j++){
      for (k=0;k<numcls[0];k++)
	fprintf(wtfile, "%e ", wt[m*numcls[0]+k]);
      fprintf(wtfile, "\n");
      m++;
    }
  }

  /*------------------------------------------------------------*/
  /*- Analysis done based on the matching weight matrix.        */
  /*------------------------------------------------------------*/
  int n0=numcls[0];
  int **codect, *clsct0;
  float **nfave;
  
  codect=(int **)calloc(n0,sizeof(int *));
  nfave=(float **)calloc(n0,sizeof(float *));
  for (i=0;i<n0;i++) {
    codect[i]=(int *)calloc(4,sizeof(int));
    nfave[i]=(float *)calloc(4,sizeof(float));
  }
  
  for (i=1,m=0;i<nbs;i++){ m+=numcls[i];}
  res=(float *)calloc(m*n0,sizeof(float));
  hdcode=(float *)calloc(m*n0,sizeof(float));
  
  //Convert weight matrix wt[] into match-split status matrix res[]
  //Summarize the reliability of all the clusters
  MatchSplit(wt, res, numcls, nbs, codect, nfave, thred);

  //Compute confidence sets and average ratios for matched/split clusters
  CLink **clist2;
  int *nsamples; 
  int **confpts, *npts;
  float *avetight, *avecov, *avejacaard, *rinclude, *csetdist;

  confpts=(int **)calloc(numcls[0],sizeof(int *));
  npts=(int *)calloc(numcls[0],sizeof(int));
  avetight=(float *)calloc(numcls[0],sizeof(float));
  avecov=(float *)calloc(numcls[0],sizeof(float));
  avejacaard=(float *)calloc(numcls[0],sizeof(float));
  rinclude=(float *)calloc(numcls[0],sizeof(float));
  csetdist=(float *)calloc(numcls[0]*numcls[0],sizeof(float));

  MatchCluster(res,numcls,nbs,thred,cls,len,&clist2, &nsamples, usesplit);
  AveConfset(clist2, numcls[0], nsamples, alpha, confpts, npts, avetight, avecov, avejacaard, rinclude, csetdist);

  //Convert match-split status matrix res[] into hard assignment code hdcode[]
  m=HardAssign(wt,res,hdcode,numcls,nbs,cls,len,thred);
  fprintf(stdout, "#bootstrap samples leading to missed component in reference: %d\n",m);

  m=AdjustHard(wt,hdcode,numcls,nbs,cls, len, thred);
  fprintf(stdout, "#bootstrap samples with split that are adjusted to different results: %d\n",m);
  
  
  /*------------------------------------------------------------*/
  /*- Output results                                            */
  /*------------------------------------------------------------*/
  //Output per cluster summary
  clsct0=(int *)calloc(n0,sizeof(int));
  for (i=0;i<len;i++) {
    if (cls[i]>=0)  {clsct0[cls[i]]++;}
  }
  for (i=0;i<n0;i++) {
    fprintf(cfile,"%d\t%d\t",i,clsct0[i]);
    for (j=0;j<4;j++)
      fprintf(cfile, "%d\t%f\t",codect[i][j],nfave[i][j]);
    //Output 1-jaccard distance=Jacaard similarity
    fprintf(cfile, "%.4f  %.3f  %.3f  %.3f", rinclude[i], avetight[i], avecov[i], 1.0-avejacaard[i]);
    fprintf(cfile, "\n");
  }

  for (i=0;i<n0;i++){
    for (j=0;j<n0;j++)
      fprintf(cfile, "%.3f ", csetdist[i*n0+j]);
    fprintf(cfile, "\n");
  }

  //Output confidence set for each reference cluster
  //Every cluster takes one row. Cluster ID, Cluster size, points IDs one by one
  for (i=0;i<n0;i++) {
    fprintf(cfile, "%d %d ",i,npts[i]);
    for (j=0;j<npts[i];j++)
      fprintf(cfile, "%d ", confpts[i][j]);
    fprintf(cfile, "\n");
  }

  //Output matching result for each bootstrap sample
  for (i=1,m=0;i<nbs;i++){
    for (j=0;j<numcls[i];j++){
      for (k=0;k<n0;k++)
	fprintf(outfile, "%8.4f ", res[m*n0+k]);
      fprintf(outfile, "\n");
      m++;
    }
  }

  //Output hard assignment result for each bootstrap sample
  for (i=1,m=0;i<nbs;i++){
    for (j=0;j<numcls[i];j++){
      for (k=0;k<n0;k++)
	fprintf(hfile, "%8.4f ", hdcode[m*n0+k]);
      fprintf(hfile, "\n");
      m++;
    }
  }

  //Summarize the per bootstrap sample matching result and print out summary
  MatchInfo(res,numcls,nbs,thred, stdout);

}


