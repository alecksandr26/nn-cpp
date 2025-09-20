#include "../include/rand.hpp"

using namespace nn::rand;

nn::rand::RandInitializer::~RandInitializer(void) = default;


template <typename T>
nn::rand::RandUniformInitializer<T>::RandUniformInitializer(T min_val, T max_val)
	: min_val_(min_val), max_val_(max_val)
{
	register_funcs();
}


template <typename T>
RandUniformInitializer<T> &nn::rand::RandUniformInitializer<T>::register_funcs(void)
{
	register_func<Mat<T> &, Mat<T> &>
		("feedforward", [this](Mat<T> & A) -> Mat<T> &{
			return A.rand_uniform(min_val_, max_val_);
		});
	return *this;
}

template class nn::rand::RandUniformInitializer<float>;
// template class nn::rand::RandUniformInitializer<double>;


template <typename T>
nn::rand::RandNormalInitializer<T>::RandNormalInitializer(T mean, T stddev)
	: mean_(mean), stddev_(stddev)
{
	register_funcs();
}

template <typename T>
RandNormalInitializer<T> &nn::rand::RandNormalInitializer<T>::register_funcs(void)
{
	register_func<Mat<T> &, Mat<T> &>
		("feedforward", [this](Mat<T> & A) -> Mat<T> &{
			return A.rand_normal(mean_, stddev_);
		});
	return *this;
}


template class nn::rand::RandNormalInitializer<float>;
// template class nn::rand::RandNormalInitializer<double>;

