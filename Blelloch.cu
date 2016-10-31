/*
A small code to test the blelloch and Hillis algrithm
Blelloch can only perform on size = 2^d array; thus a artitary array with size N should seperate into 2 arrary ->
one is 2^d(cloest to N for maximum efficiency) with Blelloch sum and another is N-2^n by using Hillis 
*/

#include<iostream>
#include <cstdlib> 
#include <cmath> 
using namespace std;
int threads = 512;
int N=1536*17;
void exclusive_cpu(double* reference,double* host)
{
	for(int i=0;i<N;i++)
	{
		for(int ii=i-1;ii>=0;ii--)
		{
			reference[i]+=host[ii];
		}
	}
}

void generate_rand(double* h)
{
	for(int i = 0;i<N;i++){
		srand (i);
		h[i]=(double(rand())/RAND_MAX-0.5)*1000;
	}
	
}
__global__ void Belloch_sum_up(double* x, int i,int Nb)
{
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	int offset = 1<<i;
	if(idx>=Nb)return;
	if(idx%offset==offset-1&&idx>=offset/2)
	{
		x[idx]+=x[idx-(offset/2)];
	}
}
__global__ void Belloch_sum_down(double* x, int i,int Nb,int d)
{
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=Nb)return;
	int offset = 1<<i;
	if(idx%offset==offset-1&&idx>=offset/2)//idx%(offset)==0
	{
		double temp=x[idx];
		x[idx]+=x[idx-offset/2];
		x[idx-offset/2]=temp;
	}
}

__global__ void Hillis_sum(double* x,double* t, int i,int Nh)
{
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=Nh)return;
	if(idx>=i)t[idx]=x[idx]+x[idx-i];
	else t[idx]=x[idx];
	
}
__global__ void shift_offset (double*a,double b,int Nh)
{
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=Nh)return;
	a[idx]+=b;
}
int main (void)
{
	cudaDeviceReset();
	cudaSetDevice(0);
	int d = int(log2(double(N)));//d
	int Nb = 1<<d;//blelloch size, 2^d
	int bs = (Nb+threads-1)/threads;
	//cout<<"Nb="<<Nb<<endl;
	double host[N];
	double reference[N];
	double result_h[N];
	double *Belloch;
	generate_rand(host);
	memset(reference,0,sizeof(double)*N); 
	exclusive_cpu(reference,host);
	
	/*Blelloch part*/	
	cudaMalloc((void**)&Belloch,sizeof(double)*Nb); 
	cudaMemcpy(Belloch,host,sizeof(double)*Nb,cudaMemcpyHostToDevice);
	for(int i=1;i<=d;i++){Belloch_sum_up<<<bs,threads>>>(Belloch,i,Nb);}
	cudaMemset(&Belloch[Nb-1],0,sizeof(double));
	for(int i=d;i>=1;i--){Belloch_sum_down<<<bs,threads>>>(Belloch,i,Nb,d);}
	/**/
	
	/*Hillis part*/	
	if(Nb!=N)
	{
		int Nh = N-Nb;//Hillis size
		double offset=0;//offset of Hillis = sum of belloch part;
		cudaMemcpy(&offset,&Belloch[Nb-1],sizeof(double),cudaMemcpyDeviceToHost);
		double *Hillis,*temp;
		cudaMalloc((void**)&Hillis,sizeof(double)*Nh); 
		cudaMalloc((void**)&temp,sizeof(double)*Nh); 
		cudaMemcpy(Hillis,&host[Nb-1],sizeof(double)*Nh,cudaMemcpyHostToDevice);
		for(int i=1;i<Nh;i*=2)
		{
			Hillis_sum<<<bs,threads>>>(Hillis,temp,i,Nh);
			cudaMemcpy(Hillis,temp,sizeof(double)*Nh,cudaMemcpyDeviceToDevice);
		}
		shift_offset<<<bs,threads>>>(Hillis,offset,Nh);	
		cudaMemcpy(&result_h[Nb],Hillis,sizeof(double)*Nh,cudaMemcpyDeviceToHost);
		cudaFree(Hillis);
		cudaFree(temp);
	}
	cudaMemcpy(result_h,Belloch,sizeof(double)*Nb,cudaMemcpyDeviceToHost);
	double diff = 0;
	
	for(int i=0;i<N;i++)diff+=(result_h[i]-reference[i])*(result_h[i]-reference[i]);
	cout<<"N\t="<<N<<"\tstd divation\t=\t"<<sqrt(diff)/N<<endl;
	cudaFree(Belloch);
	
}
