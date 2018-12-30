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
  char wtfilename[100], cfilename[100], hfilename[100];
  //char intooutfilename[100];
  FILE *infile, *outfile, *parfile, *wtfile, *cfile, *hfile/*, *intooutfile*/;
  int i,j,k,m,n,iter;
  int nbs,len,minsz=0;
  int *cls;
  float *wt, *res, *hdcode, *dist;
  int *numcls;
  float thred=0.8;
  int equalcls=0;
  
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
      case 'e':
        equalcls=1;
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
    /*- Iteration for nbs times                                  -*/
    /*------------------------------------------------------------*/
    
  float *avedist;

  avedist=(float *)calloc(nbs,sizeof(float));
  for (iter=1;iter<nbs;iter++) {
    // Insert the bootstrap sample as the current reference
    for(m=0;m<len;m++) {
      cls[m] = cls[len*iter+m];
    }
    
    /*------------------------------------------------------------*/
    /*- Alignment done here to get the key matching matrix wt[]  -*/
    /*------------------------------------------------------------*/
    
    align(cls,nbs,len,&wt,&numcls,&dist,equalcls);
  
    /*------------------------------------------------------------*/
    /*- Output the weight matrix and the distances between        */
    /*- clustering results.                                       */
    /*------------------------------------------------------------*/
    for (i=1,avedist[iter]=0.0;i<nbs;i++){
      avedist[iter]+=dist[i];
    }
    avedist[iter]/=(float)nbs; //v1 is the average Distance to the other bootstrap samples

    free(wt);
    free(numcls);
    free(dist);
  }

  float v1;
  v1=avedist[1]; k=1;
  for (iter=1;iter<nbs;iter++) {
    if (v1>avedist[iter]) {
      v1=avedist[iter];
      k=iter; //k is the id for bootstrap sample that should be the reference
    }
  }
  fprintf(stdout, "%d\n",k);


}
