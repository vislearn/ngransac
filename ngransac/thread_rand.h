#pragma once

#include <random>

/** Classes and methods for generating random numbers in multi-threaded programs. */

/**
 * @brief Provides random numbers for multiple threads.
 * 
 * Singelton class. Holds a random number generator for each thread and gives random numbers for the current thread.
 */
class ThreadRand
{
public:
  /**
   * @brief Returns a random integer (uniform distribution).
   * 
   * @param min Minimum value of the random integer (inclusive).
   * @param max Maximum value of the random integer (exclusive).
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return int Random integer value.
   */
  static int irand(int min, int max, int tid = -1);
  
  /**
   * @brief Returns a random double value (uniform distribution).
   * 
   * @param min Minimum value of the random double (inclusive).
   * @param max Maximum value of the random double (inclusive).
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random double value.
   */
  static double drand(double min, double max, int tid = -1);
  
  /**
   * @brief Returns a random double value (Gauss distribution).
   * 
   * @param mean Mean of the Gauss distribution to sample from.
   * @param stdDev Standard deviation of the Gauss distribution to sample from.
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random double value.
   */
  static double dgauss(double mean, double stdDev, int tid = -1);
    
  /**
   * @brief Re-Initialize the object with the given seed.
   * 
   * @param seed Seed to initialize the random number generators (seed is incremented by one for each generator).
   * @return void
   */
  static void forceInit(unsigned seed);
  
  /**
   * @brief List of random number generators. One for each thread.
   * 
   */
  static std::vector<std::mt19937> generators;

  /**
   * @brief Initialize class with the given seed.
   * 
   * Method will create a random number generator for each thread. The given seed 
   * will be incremented by one for each generator. This methods is automatically 
   * called when this calss is used the first time.
   * 
   * @param seed Optional parameter. Seed to be used when initializing the generators. Will be incremented by one for each generator.
   * @return void
   */
  static void init(unsigned seed = 1305);

private:  

  /**
   * @brief True if the class has been initialized already
   */
  static bool initialised;

};

/**
  * @brief Returns a random integer (uniform distribution).
  * 
  * This method used the ThreadRand class.
  * 
  * @param min Minimum value of the random integer (inclusive).
  * @param max Maximum value of the random integer (exclusive).
  * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
  * @return int Random integer value.
  */
int irand(int incMin, int excMax, int tid = -1);
/**
  * @brief Returns a random double value (uniform distribution).
  * 
  * This method used the ThreadRand class.
  * 
  * @param min Minimum value of the random double (inclusive).
  * @param max Maximum value of the random double (inclusive).
  * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
  * @return double Random double value.
  */
double drand(double incMin, double incMax, int tid = -1);

  /**
   * @brief Returns a random integer value (Gauss distribution).
   * 
   * This method used the ThreadRand class.
   * 
   * @param mean Mean of the Gauss distribution to sample from.
   * @param stdDev Standard deviation of the Gauss distribution to sample from.
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random integer value.
   */
int igauss(int mean, int stdDev, int tid = -1);

  /**
   * @brief Returns a random double value (Gauss distribution).
   * 
   * This method used the ThreadRand class.
   * 
   * @param mean Mean of the Gauss distribution to sample from.
   * @param stdDev Standard deviation of the Gauss distribution to sample from.
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random double value.
   */
double dgauss(double mean, double stdDev, int tid = -1);
