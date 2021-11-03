#if defined(_OPENMP)
#include <omp.h>
#endif
#include "thread_rand.h"

std::vector<std::mt19937> ThreadRand::generators;
bool ThreadRand::initialised = false;

void ThreadRand::forceInit(unsigned seed)
{
    initialised = false;
    init(seed);
}

void ThreadRand::init(unsigned seed)
{
    #pragma omp critical
    {
	if(!initialised)
	{
#if defined(_OPENMP)
	    unsigned nThreads = omp_get_max_threads();
#else
	    unsigned nThreads = 1;
#endif
	    
	    for(unsigned i = 0; i < nThreads; i++)
	    {    
		generators.push_back(std::mt19937());
		generators[i].seed(i+seed);
	    }

	    initialised = true;
	}    
    }
}

int ThreadRand::irand(int min, int max, int tid)
{
    std::uniform_int_distribution<int> dist(min, max);
#if defined(_OPENMP)
    unsigned threadID = omp_get_thread_num();
#else
    unsigned threadID = 0;
#endif
    if(tid >= 0) threadID = tid;
    
    if(!initialised) init();
  
    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::drand(double min, double max, int tid)
{
    std::uniform_real_distribution<double> dist(min, max);
    
#if defined(_OPENMP)
    unsigned threadID = omp_get_thread_num();
#else
    unsigned threadID = 0;
#endif
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::dgauss(double mean, double stdDev, int tid)
{
    std::normal_distribution<double> dist(mean, stdDev);
    
#if defined(_OPENMP)
    unsigned threadID = omp_get_thread_num();
#else
    unsigned threadID = 0;
#endif
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

int irand(int incMin, int excMax, int tid)
{
    return ThreadRand::irand(incMin, excMax - 1, tid);
}

double drand(double incMin, double incMax,int tid)
{
    return ThreadRand::drand(incMin, incMax, tid);
}

int igauss(int mean, int stdDev, int tid)
{
    return (int) ThreadRand::dgauss(mean, stdDev, tid);
}

double dgauss(double mean, double stdDev, int tid)
{
    return ThreadRand::dgauss(mean, stdDev, tid);
}
