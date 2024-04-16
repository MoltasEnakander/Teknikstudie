#include <stdio.h>
#include <chrono>
#include <ctime>

/*__global__
void saxpy(int n, float a, float* x, float* y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
	int N = 2048;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;

    std::cout << "elapsed: " << elapsed.count() << "s\n";

	

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = max(maxError, abs(y[i] - 4.0f));
	}
	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}*/


__global__ void child_k(const int i, const int j, const int FRAMES, const int NUM_VIEWS, float* summedSignals, const int blockid)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockid == 1){
		id += 32 * 32 * FRAMES;
	}
	//i = i % 32;
	id += (i + j * min(32, NUM_VIEWS)) * FRAMES;
	summedSignals[id] = id;
}

__global__ void parent_k(const int FRAMES, const int NUM_VIEWS, float* summedSignals)
{
	int i = /*blockIdx.x * blockDim.x +*/ threadIdx.x;
	int j = /*blockIdx.y * blockDim.y +*/ threadIdx.y;
	int stop = NUM_VIEWS * NUM_VIEWS - 1024;

	if (blockIdx.x == 1 && threadIdx.x + threadIdx.y * blockDim.x >= stop){
		return;
	}

	child_k<<<(FRAMES+255)/256, 256>>>(i, j, FRAMES, NUM_VIEWS, summedSignals, blockIdx.x);

	/*for (int k = 0; k < FRAMES; ++k)
	{		
		int id = k + (i + j * 32) * FRAMES;
		if (blockIdx.x == 1){
			id += 32 * 32 * FRAMES;
		}
		summedSignals[id] = id;
	}*/
}

int main(void)
{
	const int NUM_VIEWS = 33;
	const int FRAMES = 512;
	const int MAX_THREADS_PER_BLOCK = 1024;
	float* summedSignals = (float*)malloc(sizeof(float) * NUM_VIEWS * NUM_VIEWS * FRAMES); // each beam will have its own signal buffer of length FRAMES
	float* d_summedSignals;
	cudaMalloc(&d_summedSignals, sizeof(float) * NUM_VIEWS * NUM_VIEWS * FRAMES);

	for (int i = 0; i < NUM_VIEWS * NUM_VIEWS * FRAMES; ++i)
	{
		summedSignals[i] = 0.0f;
	}

	cudaMemcpy(d_summedSignals, summedSignals, sizeof(float) * NUM_VIEWS * NUM_VIEWS * FRAMES, cudaMemcpyHostToDevice);

	int numBlocks;
	dim3 threadsPerBlock;
	if (NUM_VIEWS * NUM_VIEWS >= MAX_THREADS_PER_BLOCK){
		numBlocks = 2;
		threadsPerBlock = dim3(32, 32);		
	}
	else{
		numBlocks = 1;
		threadsPerBlock = dim3(NUM_VIEWS, NUM_VIEWS);
	}
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	parent_k<<<numBlocks, threadsPerBlock>>>(FRAMES, NUM_VIEWS, d_summedSignals);
	cudaDeviceSynchronize();
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed = end-start;

	printf("elapsed: %f s \n", elapsed.count());
	cudaMemcpy(summedSignals, d_summedSignals, sizeof(float) * NUM_VIEWS * NUM_VIEWS * FRAMES, cudaMemcpyDeviceToHost);	

	float error = 0.0f;
	for (int i = 0; i < NUM_VIEWS * NUM_VIEWS * FRAMES; ++i)
	{
		error += summedSignals[i] - i;
		//printf("e: %f %f %d \n", summedSignals[i] - i, summedSignals[i], i);
	}

	printf("error: %f\n", error);

	cudaFree(d_summedSignals);
	free(summedSignals);
}