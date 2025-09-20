#ifndef NN_MODEL_INCLUDED
#define NN_MODEL_INCLUDED

#include "utils.hpp"
#include "mat.hpp"

namespace nn::models {
	using namespace utils;
	using namespace mathops;
	
	class Model : protected GenericVTable {
	public:
		using GenericVTable::GenericVTable;
		
		virtual ~Model(void) = 0;
		
		template <typename T>
		Mat<T> operator()(const Mat<T> &X)
		{
			return get_func<Mat<T>, const Mat<T> &>("feedforward")(X);
		}
		
	};
}

#endif

