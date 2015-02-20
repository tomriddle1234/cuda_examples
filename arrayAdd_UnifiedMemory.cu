#include <string.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)){
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)){
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0){
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else{
		//  Handle error
		return 0;
	}
}

//  Posix/Linux
#else
#include <sys/time.h>
double get_wall_time(){
	struct timeval time;
	if (gettimeofday(&time, NULL)){
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif

static void HandleError(cudaError_t err,
	const char *file,
	int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


const int arraySize = 1000000;
struct LargeData
{
	int hugevalue[arraySize] ;
};

int cc[arraySize];
int aa[arraySize];
int bb[arraySize];

void cpuAdd()
{
	//start cpu computation
	printf("Start CPU \n");

	for (size_t i = 0; i < arraySize; ++i)
	{
		aa[i] = i;
		bb[i] = i + 1;
	}

	double wall_time0, wall_time1;
	double cpu_time0, cpu_time1;

	wall_time0 = get_wall_time();
	cpu_time0 = get_cpu_time();

	for (size_t i = 0; i < arraySize; ++i)
		cc[i] = aa[i] + bb[i];

	wall_time1 = get_wall_time();
	cpu_time1 = get_cpu_time();

	printf("=== CPU ===\n");
	printf("CPU -- Wall time: %3.10f ms \n", (wall_time1 - wall_time0) * 1000);
	printf("CPU -- Cpu time: %3.10f ms \n", (cpu_time1 - cpu_time0) * 1000);
}

__global__
void addall(LargeData * a, LargeData * b, LargeData * c)
{
	int i = threadIdx.x;
	int j = blockIdx.x ;
	c->hugevalue[j * blockDim.x + i] = a->hugevalue[j * blockDim.x + i] + b->hugevalue[j * blockDim.x + i];
}


void launchAdd(LargeData * a, LargeData * b, LargeData * c)
{
	float time;
	double wall_time0, wall_time1;
	double cpu_time0, cpu_time1;
	cudaEvent_t start, stop;

	dim3 threadsPerBlock(256, 1);
	dim3 numBlocks(arraySize / threadsPerBlock.x + 1, 1);

	wall_time0 = get_wall_time();
	cpu_time0 = get_cpu_time();

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	addall << <numBlocks, threadsPerBlock >> >(a, b, c);
	cudaDeviceSynchronize();

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

	wall_time1 = get_wall_time();
	cpu_time1 = get_cpu_time();

	printf("=== CUDA Execution Time: ===\n");
	printf("Cuda event time to generate:  %3.10f ms \n", time);
	printf("Wall time: %3.10f ms \n", (wall_time1 - wall_time0) * 1000);
	printf("Cpu time: %3.10f ms \n", (cpu_time1 - cpu_time0) * 1000);
}

int main(void)
{

	cpuAdd();
	LargeData *a;
	LargeData *b;
	LargeData *c;

	cudaMallocManaged((void**)&a, sizeof(LargeData));
	cudaMallocManaged((void**)&b, sizeof(LargeData));
	cudaMallocManaged((void**)&c, sizeof(LargeData));

	for (size_t i = 0; i < arraySize; ++i)
	{
		a->hugevalue[i] = i;
		b->hugevalue[i] = i+1;
		c->hugevalue[i] = 0;
	}
	
	launchAdd(a, b, c);

	for (size_t i = 0; i < 10; ++i)
		printf("%d,", c->hugevalue[i]);
	printf("\n");
	printf("%d\n", a->hugevalue[arraySize - 2]);
	printf("%d\n", b->hugevalue[arraySize - 2]);
	printf("%d\n" ,c->hugevalue[arraySize-2]);
	printf("%d\n", c->hugevalue[255]);
	printf("%d\n", c->hugevalue[256]);
	printf("CUDA Finished.\n");

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	cudaDeviceReset();

	return 0;
}
