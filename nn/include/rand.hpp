#ifndef NN_RAND_INCLUDED
#define NN_RAND_INCLUDED

#include "mat.hpp"
#include "utils.hpp"

namespace nn::rand {
	using namespace mathops;
	using namespace utils;
	
	class RandInitializer : protected GenericVTable {
	public:
		using GenericVTable::GenericVTable;
		
		virtual ~RandInitializer(void) = 0;

		template <typename T>
		Mat<T> &operator()(Mat<T> &A)
		{
			get_func<Mat<T> &, Mat<T> &>("feedforward", __FILE__, __LINE__)(A);
			return A;
		}
	};
	
	template <typename T>
	class RandUniformInitializer : public RandInitializer {
	public:
		using RandInitializer::RandInitializer;

		RandUniformInitializer(T min_val = static_cast<T>(-1.0f), T max_val = static_cast<T>(1.0f));
		
	private:
		RandUniformInitializer &register_funcs(void) override;

		T min_val_;
		T max_val_;
	};

	template <typename T>
	class RandNormalInitializer : public RandInitializer {
	public:
		using RandInitializer::RandInitializer;
		
		RandNormalInitializer(T mean = static_cast<T>(0.0f), T stddev = static_cast<T>(1.0f));
		
	private:
		RandNormalInitializer &register_funcs(void) override;

		T mean_;
		T stddev_;
	};
}


#endif
