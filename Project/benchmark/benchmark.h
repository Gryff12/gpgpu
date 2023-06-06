//
// Created by Maxime on 02/06/2023.
//

#ifndef PROJECT_TIME_H
#define PROJECT_TIME_H

#include <functional>

#ifndef __CUDA_ARCH__
typedef bool __nv_bool;
#endif

namespace benchmark {
	template <typename T>
	struct time_output {
	public:
		T result;
		double ms;
	};

	template<typename T>
	time_output<T> timeit(std::function<T(void)> func) {

		// Build chrono objects
		using std::chrono::high_resolution_clock;
		using std::chrono::duration_cast;
		using std::chrono::duration;
		using std::chrono::milliseconds;

		// Measure time
		auto t1 = high_resolution_clock::now(); // Start time
		T result = func();
		auto t2 = high_resolution_clock::now(); // End time

		duration<double, std::milli> ms_double = t2 - t1; // Time delta

		// Build result struct
		time_output<T> output;
		output.result = result;
		output.ms = ms_double.count();

		return output;
	}
}

#define TIMEIT(T, func) benchmark::timeit<T>( []() { return func ;})

#endif //PROJECT_TIME_H
