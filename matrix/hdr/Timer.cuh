#ifndef __TIMER_H__
#define __TIMER_H__

class Timer
{
	cudaEvent_t begin, end;
public:
	Timer()
	{
		cudaEventCreate(&begin);
		cudaEventCreate(&end);
	}

	~Timer()
	{
		cudaEventDestroy(begin);
		cudaEventDestroy(end);
	}

	void start()
	{
		cudaEventRecord(begin, 0);
	}

	float stop()
	{
		float elapse;
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapse, begin, end);
		return elapse;
	}

};

#endif
