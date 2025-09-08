#ifndef NN_MODEL_INCLUDED
#define NN_MODEL_INCLUDED

#include "mat.hpp"

namespace nn::models {
	using namespace mathops;
	
	template <typename T>
	class Model {
	public:
		virtual Mat<T> operator()(const Mat<T> &X) = 0;
	};
}

#endif




