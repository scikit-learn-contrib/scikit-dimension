#include "mex.h"
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string.h>


long num;
long dim;
long dim_right;
double CORR_DIM;
double DIM;
double TAK_DIM;
long** sp_index;
long* nr_elements;

long ctr=4;
long ctr2=0;



/* uniform [0,1] random number generator
   developed by Pierre Lecuyer based on a clever
   and tested combination of two linear congruential
   sequences 
*/

/*
s1 and s2 are the seeds (nonnegative integers)
*/

double uni()
{
  static long s1 = 55555;
  static long s2 = 99999;
  static double factor = 1.0/2147483563.0;
  register long k,z;
  k= s1 /53668;
  s1 =40014*(s1%53668)-k*12211;
  if (s1 < 0) s1 += 2147483563;
  k=s2/52774;
  s2=40692*(s2%52774)-k*3791;
  if (s2 < 0) s2 += 2147483399;

  /*
  z = abs(s1 ^ s2);
  */
  z= (s1 - 2147483563) + s2;
  if (z < 1) z += 2147483562;

  return(((double)(z))*factor);
}



double comp_slope(double* x,double* y,long num)
{
  double avg_x=0, avg_y=0, slope=0,enu=0,deno=0, w_tot=0;
  long i;
  double* w=new double[long(num)];

  for(i=0;i<num;i++)
   { w[i]=1.0/(1.0*(num-i)); w_tot+=w[i]; }
  for(i=0;i<num;i++)
   { avg_x+=w[i]*x[i]; avg_y+=w[i]*y[i];}

  avg_x/=w_tot; avg_y/=w_tot;

  for(i=0;i<num;i++)
  {
    enu  +=w[i]*x[i]*(y[i]-avg_y);
    deno +=w[i]*x[i]*(x[i]-avg_x);
  }
  slope=enu/deno;
  delete w;
  return slope;
}



// estimates the intrinsic dimension of the dim-dimensional data from num samples
// max_dim: is the maximal possible dimension to be estimated
// if TAKENS==1, CORR==1 then also the Takens estimator and the correlation dimension estimator are calculated
// SPARSE=1 if SPARSE data is used

void Calc_Dimension(double** data, long dim, long max_dim, double num, char TAKENS,char CORR,char SPARSE)
{
  long i,j,k,l,m,counter;

  double dist;
  double* min_dist=new double[long(num)];
  for(i=0;i<num;i++)
   min_dist[i]=1E10;

  double* norm2;
  if(SPARSE)
  {
    norm2 = new double[long(num)];
    for(i=0;i<num;i++)
    {
      norm2[i]=0;
      for(k=0;k<nr_elements[i];k++) norm2[i]+=data[i][k]*data[i][k];
    }
  }

  /*time_t anf; time_t end;
  anf=time(NULL);
  struct time t1,t2;
  gettime(&t1);  */


  // compute the one-nearest neighbor distances of each point
  for(i=0;i<num;i++)
  {
    for(j=i+1;j<num; j++)
    {
      // compute the squared distance to this point
      dist=0;
      if(!SPARSE)
      {
        for(k=0;k<dim;k++)
        {
          dist += (data[i][k]-data[j][k])*(data[i][k]-data[j][k]);
          if(dist>min_dist[i] && dist>min_dist[j])
           { k=dim; }
        }
      }
      else
      {
        dist = (norm2[i]+norm2[j]);
        counter=0;
        if(nr_elements[i]<nr_elements[j])
        {
          for(k=0;k<nr_elements[i];k++)
          {
            while(sp_index[j][counter]<sp_index[i][k]) counter++;
            if(sp_index[i][k]==sp_index[j][counter])
             dist-=2.0*data[i][k]*data[j][counter];
          }
        }
        else
        {
          for(k=0;k<nr_elements[j];k++)
          {
            while(sp_index[i][counter]<sp_index[j][k]) counter++;
            if(sp_index[j][k]==sp_index[i][counter])
             dist-=2.0*data[j][k]*data[i][counter];
          }
        }
        if(dist<0) dist=0;
      }
      if(dist < min_dist[i])  // check if it is smaller than the maximal minimal distance
         min_dist[i]=dist;
      if(dist < min_dist[j])
         min_dist[j]=dist;
    }
  }



  // compute mean minimal distance
  double avg_min=0;
  double min_min=1E+10;
  double max_min=0;
  for(i=0;i<num;i++)
  {
    avg_min+=sqrt(min_dist[i]);
    if(min_dist[i]<min_min)
     min_min=min_dist[i];
    if(min_dist[i]>max_min)
     max_min=min_dist[i];
  }
  max_min=sqrt(max_min);
  min_min=sqrt(min_min);
  avg_min/=num;

  // compute variance of the minimal distance
  double std_min=0;
  for(i=0;i<num;i++)
   std_min+=pow(sqrt(min_dist[i])-avg_min,2);
  std_min/=num;
  std_min=sqrt(std_min);

  long NR_EST=5;

  double TAKENS_SCALE=0;                 // the scale (upper bound on the distances) used in the Takens estimator
  double* CORR_SCALE=new double[NR_EST];

  for(k=0;k<5;k++)
  {
    CORR_SCALE[k] =avg_min + k*0.2*std_min;
    CORR_SCALE[k]*=CORR_SCALE[k];
  }
  TAKENS_SCALE=avg_min+std_min;
  TAKENS_SCALE*=TAKENS_SCALE;

  // cut_dist is defined as the mean minimal distance
  double cut_dist=avg_min;  //avg_min+0*std_min;
  double cut_dist_sq=pow(cut_dist,2);


  // now the final algorithm starts
  // we assume a form h ~ (log n/n)^1/d, this guarantees nh^d -> infinity and h -> 0

  double scale=0;
  double** h=new double*[max_dim];
  for(i=0;i<max_dim;i++)
   h[i]=new double[NR_EST];

  double** h_squared=new double*[max_dim];
  for(i=0;i<max_dim;i++)
   h_squared[i]=new double[NR_EST];

  double*** est=new double**[max_dim];
  for(i=0;i<max_dim;i++)
  {
    est[i]=new double*[NR_EST];
    for(j=0;j<NR_EST;j++)
     est[i][j]=new double[NR_EST*NR_EST];
  }

  double TAKENS_EST=0;
  long TAKENS_NR=0;
  double* CORR_EST=new double[NR_EST];
  for(i=0;i<NR_EST;i++) CORR_EST[i]=0;

  long* divisions=new long[NR_EST];
  for(i=0;i<NR_EST;i++)
   divisions[i]=NR_EST-i;
  //long divisions[5]={10,5,3,2,1};
  double* size=new double[NR_EST];
  for(i=0;i<NR_EST;i++)
   size[i]=floor(num/(1.0*divisions[i]));


  // initialize the scale of each dimension and the corresponding h
  for(m=1;m<=max_dim;m++)
  {
    for(k=0;k<NR_EST;k++)                      // we have NR_EST partitions of the data
    {
      h[m-1][k]=cut_dist*powl((num*log(size[k]))/(size[k]*log(num)),1.0/m);
      h_squared[m-1][k]=h[m-1][k]*h[m-1][k];
      for(j=0;j<NR_EST*NR_EST;j++)
       est[m-1][k][j]=0;
    }
  }

  double min_scale=h_squared[0][0];
  long index_k_x; long index_k_y;

  for(i=0;i<num;i++)
  {
    for(j=i+1;j<num;j++)
    {
      dist=0;
      if(!SPARSE)
      {
        for(l=0;l<dim;l++)
        {
          dist += (data[i][l]-data[j][l])*(data[i][l]-data[j][l]);
          if(dist>min_scale && (!TAKENS || dist > TAKENS_SCALE) && (!CORR || dist > CORR_SCALE[4]))
           l=dim;
        }
      }
      else
      {
        dist = (norm2[i]+norm2[j]);
        counter=0;
        if(nr_elements[i]<nr_elements[j])
        {
          for(k=0;k<nr_elements[i];k++)
          {
            while(sp_index[j][counter]<sp_index[i][k]) counter++;
            if(sp_index[i][k]==sp_index[j][counter])
             dist-=2.0*data[i][k]*data[j][counter];
          }
        }
        else
        {
          for(k=0;k<nr_elements[j];k++)
          {
            while(sp_index[i][counter]<sp_index[j][k]) counter++;
            if(sp_index[j][k]==sp_index[i][counter])
             dist-=2.0*data[j][k]*data[i][counter];
          }
        }
        //dist*=2.0;
        if(dist<0) dist=0;
      }
      if(TAKENS && dist<TAKENS_SCALE && dist>0 )
        { TAKENS_EST+=0.5*log(dist/TAKENS_SCALE);  TAKENS_NR++; }
      if(dist<min_scale)
      {
        if(CORR)
        {
          for(k=0;k<5;k++)
          {
            if(dist<CORR_SCALE[k]) CORR_EST[k]++;
          }
        }
        for(k=0;k<NR_EST;k++)
        {
          index_k_x=i % divisions[k];
          index_k_y=j % divisions[k];
          for(m=1;m<=max_dim;m++)  // try all dimensions from 1 to max_dim
          {
            if(dist < h_squared[m-1][k])
            {
              if(index_k_x<index_k_y)
                est[m-1][k][index_k_x*divisions[k]+index_k_y]+=1.0-dist/h_squared[m-1][k];
              else
                est[m-1][k][index_k_y*divisions[k]+index_k_x]+=1.0-dist/h_squared[m-1][k];
            }
            else m=max_dim;  // note that h_squared is monotonically decreasing with m
                             // which means that if the condition dist < h_squared[m-1][k]  does not
                             // hold for m it will also not hold for any larger m
          }
        }
      }
    }
  }
  long double total=0;
  for(m=1;m<=max_dim;m++)
  {
    for(k=0;k<NR_EST;k++)
    {
      total=0;
      for(i=0;i<divisions[k];i++)
      {
        for(j=0;j<divisions[k];j++)
        {
          if(i==j)
          {
            if(i < long(num) % divisions[k])
             total+=est[m-1][k][i*divisions[k]+j]/(0.5*(size[k]+1)*size[k]);//*pow(h[m-1][k],m));
            else
             total+=est[m-1][k][i*divisions[k]+j]/(0.5*size[k]*(size[k]-1));//*pow(h[m-1][k],m));
          }
          else
          {
            if(i < long(num) % divisions[k] && j < long(num) % divisions[k])
             total+=est[m-1][k][i*divisions[k]+j]/2.0/(0.5*(size[k]+1)*(size[k]+1));//*pow(h[m-1][k],m));
            else
            {
               if(i>=long(num) % divisions[k] && j >= long(num) % divisions[k])
                total+=est[m-1][k][i*divisions[k]+j]/2.0/(0.5*size[k]*size[k]);//*pow(h[m-1][k],m));
               else
                total+=est[m-1][k][i*divisions[k]+j]/2.0/(0.5*(size[k]+1)*size[k]);//*pow(h[m-1][k],m));
            }
          }
          /*if(i==j)
           total+=est[m-1][k][i*divisions[k]+j]/(0.5*size[k]*(size[k]-1));//*pow(h[m-1][k],m));
          else
           total+=est[m-1][k][i*divisions[k]+j]/2.0/(0.5*size[k]*size[k]);//*pow(h[m-1][k],m)); */
        }
      }
      est[m-1][k][0]=2.0*total/(divisions[k]*(divisions[k]+1));
    }
  }

  if(TAKENS)
  {
    TAKENS_EST/=TAKENS_NR;//size[4]*(size[4]-1);
    TAKENS_EST=-1/TAKENS_EST;
    TAK_DIM=TAKENS_EST;
    //mexPrintf("Takens estimate: %f\n",TAKENS_EST);
  }
  if(CORR)
  {
    for(k=0;k<5;k++)
    {
      CORR_EST[k]/=0.5*size[NR_EST-1]*(size[NR_EST-1]-1);
      CORR_EST[k]=log(CORR_EST[k]);
      CORR_SCALE[k]=log(sqrt(CORR_SCALE[k]));
    }
    CORR_DIM=comp_slope(CORR_SCALE,CORR_EST,5);
    //mexPrintf("Correlation dimension: %f\n",CORR_DIM);
  }
  double minimum=100000;
  double* slope_est=new double[max_dim];
  for(m=1;m<=max_dim;m++)
  {
    for(k=0;k<NR_EST;k++)
    {
      CORR_EST[k]=log(est[m-1][k][0]) - m*log(h[m-1][k]);  // note that we do here the normalization with 1/h^m
                                                           // which leads to the additional - m log h factor
                                                           // the direct way would lead to numerical problems
      CORR_SCALE[k]=log(h[m-1][k]);
    }
    slope_est[m-1]=comp_slope(CORR_SCALE,CORR_EST,NR_EST);
    if(fabs(slope_est[m-1])<minimum) { minimum=slope_est[m-1]; DIM=m; }
    //mexPrintf("Slope of Dimension: %d, %f, %f\n",m,slope_est[m-1],slope_est[m-1]+m);
  }
  //mexPrintf("Intrinsic dim. estimate: %d\n",min_est);

  /*end=time(NULL);
  gettime(&t2);
  double est_time;

  if(t2.ti_hund - t1.ti_hund < 0) { t1.ti_sec=t1.ti_sec+1; est_time=(100.0+(t2.ti_hund - t1.ti_hund))/100.0;}
  else {est_time=(t2.ti_hund - t1.ti_hund)/100.0;}

  if(t2.ti_sec - t1.ti_sec < 0) { t1.ti_min=t1.ti_min+1; est_time+=60.0+(t2.ti_sec - t1.ti_sec);}
  else {est_time+=t2.ti_sec - t1.ti_sec;}

  if(t2.ti_min - t1.ti_min < 0) { est_time+=60.0*(60.0-(t2.ti_min-t1.ti_min)); }
  else {est_time +=60.0*(t2.ti_min-t1.ti_min);}
  mexPrintf("Minimal Scale: %f,   Time: %1.2f seconds \n",h[0][4],est_time);  */
  mexPrintf("Minimal Scale: %f \n",h[0][4]);


  // free memory
  delete min_dist, CORR_SCALE, CORR_EST, divisions, size, slope_est;
  for(i=0;i<max_dim;i++)
   { delete h[i]; delete h_squared[i]; for(j=0;j<NR_EST;j++) delete est[i][j]; delete est[i];}
  delete h, h_squared, est;
  if(SPARSE) delete norm2;
}

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  // input is the data matrix where one column is one datapoint
  // the matrix can be either sparse or full
  // output is the estimate of the intrinsic dimension of the data
  // as defined in
  //
  // M. Hein, J-Y. Audibert
  // Intrinsic dimensionality estimation of submanifolds in Euclidean space
  // Proceedings of the 22nd ICML, 289-296, Eds. L. de Raedt and S. Wrobel, 2005
  //
  // Please cite this paper if you use the estimator

  double mean_dim;
  char TAKENS=1; // 1: calculate also the Takens estimate
  char CORR  =1; // 1: calculate the correlation dimension
  long i,j;

  long max_dim;
  int* row_ind,* cul_col_ind;

  // the data vector
  double** data;

  long num_read;

  double* dim_est, *dim_est_corr, *dim_est_tak;

  char SPARSE=0;

  // check arguments
  if (nrhs != 1)
  {
    mexWarnMsgTxt("Usage: K = get_Dim_Est(X)");
    return;
  }

  if(!mxIsNumeric(prhs[0])) { mexWarnMsgTxt("No cell or structure array data as input !"); return; }
  if(mxIsComplex(prhs[0]))  { mexWarnMsgTxt("No complex data as input !"); return; }

  if(mxIsSparse(prhs[0])) // case 1: sparse data matrix
  {
    mexPrintf("Sparse matrix detected ...\n");
    SPARSE=1;
    // get Parameters from the input matrix
    dim = mxGetM(prhs[0]);  // number of rows = dimension
    num = mxGetN(prhs[0]);  // number of columns = number of datapoints
    row_ind= mxGetIr(prhs[0]);      // row number of the corresponding nonzero entry in the array points
    cul_col_ind= mxGetJc(prhs[0]);  // cumulative number of nonzero columns
    double* points = mxGetPr(prhs[0]);

    // SPARSE representation
    sp_index=new long*[long(num)];   // indices of the nonzero entries of each data vector
    nr_elements=new long[long(num)]; // number of nonzero entries in each data vector

    // determine the number of nonzero entries for each data point
    //for(i=0; i<=num; i++)
    // mexPrintf("%i\n",cul_col_ind[i]);
    nr_elements[0]=cul_col_ind[1];
    for(i=2;i<num+1;i++)
      nr_elements[i-1]=cul_col_ind[i]-cul_col_ind[i-1];

    // initialize the data matrix and sp_index
    data=new double*[long(num)];
    for(i=0;i<num;i++)
    {
      sp_index[i]=new long[nr_elements[i]];
      data[i]=new double[nr_elements[i]];
      for(j=0;j<nr_elements[i];j++)
      {
        sp_index[i][j]=row_ind[cul_col_ind[i]+j];
        data[i][j]=points[cul_col_ind[i]+j];
      }
    }
  }
  else                    // case 2: full matrix
  {
    SPARSE=0;
    // get Parameters from the input matrix
    dim = mxGetM(prhs[0]);  // number of rows = dimension
    num = mxGetN(prhs[0]);  // number of columns = number of datapoints
    double* points = mxGetPr(prhs[0]);

    // allocate memory (this is inefficient and will be optimized in a future version)
    data = new double*[long(num)];
    for(i=0;i<num;i++)
      data[i]=new double[dim];

    // initialize data
    int counter=0;
    for(i=0;i<num;i++)
    {
      for(j=0;j<dim;j++)
       { data[i][j]=points[counter]; counter++; }
    }
  }

  if(num<10)  { mexWarnMsgTxt("The algorithm needs at least 10 data points !"); return; }
  // give feedback to the user
  mexPrintf("Dimension of the data: %i, Number of data points: %i\n",dim,num);

  max_dim=dim;
  long runs=1;
  dim_est=new double[runs];
  dim_est_corr=new double[runs];
  dim_est_tak=new double[runs];

  long right=0,corr_right=0,tak_right=0;
  double CORR_DIM_MEAN=0,CORR_DIM_MEAN2=0;
  double TAK_DIM_MEAN=0, TAK_DIM_MEAN2=0;
  for(ctr=0;ctr<runs;ctr++)
  {
     if(max_dim>dim)
      { max_dim=dim; }
     //if(max_dim>500) max_dim=500;

     Calc_Dimension(data, dim, max_dim, num,TAKENS,CORR,SPARSE);

     dim_est[ctr]=DIM; // Intrinsic Dim. Estimate

     dim_est_corr[ctr]=ceil(CORR_DIM-0.5);
     dim_est_tak[ctr]=ceil(TAK_DIM-0.5);

     CORR_DIM_MEAN+=CORR_DIM;
     TAK_DIM_MEAN+=TAK_DIM;

     mexPrintf("\n");
     mexPrintf("Direct Estimates:\n");
     mexPrintf("-----------------\n");
     mexPrintf("Intrinsic Dim. Estimate: %1.3f, Correlation Dimension:%1.3f, Takens Estimator:%1.3f\n",dim_est[ctr],CORR_DIM,TAK_DIM);
     mexPrintf("\n");
     mexPrintf("Rounded Estimates:\n");
     mexPrintf("------------------\n");
     mexPrintf("Intrinsic Dim. Estimate: %1.0f, Correlation Dimension:%1.0f, Takens Estimator:%1.0f\n",dim_est[ctr],dim_est_corr[ctr],dim_est_tak[ctr]);
     mexPrintf("\n");

  }

  for(j=0;j<num;j++)
    delete data[j];
  delete data;

  delete dim_est, dim_est_corr, dim_est_tak;
  if(SPARSE)
  {
    for(i=0;i<num;i++) delete sp_index[i];
    delete sp_index;
    delete nr_elements;
  }
  plhs[0] = mxCreateDoubleMatrix(1, 3, mxREAL);
  double* pointer=mxGetPr(plhs[0]);
  pointer[0]=DIM;
  pointer[1]=CORR_DIM;
  pointer[2]=TAK_DIM;
}


