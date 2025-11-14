#include "../include/nn.hpp"
#include "../include/activation_func.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

using namespace nn::activation_funcs;
using namespace nn::models;

template <typename T>
WeightedModel<T>::~WeightedModel(void) = default;


template <typename T>
Mat<T> WeightedModel<T>::test(const std::shared_ptr<std::vector<Mat<T>>> X_test, const std::shared_ptr<std::vector<Mat<T>>> Y_test)
{
	loss_->set_inputs(X_test);
	loss_->set_outputs(Y_test);
	
	// Run the loss function
	// TODO: Should return the captured metrics, the metrics could use the returned output, of loss
	return (*loss_)();
}

template <typename T>
WeightedModel<T> &WeightedModel<T>::set_loss(std::shared_ptr<Loss<T>> loss)
{
	loss_ = loss;
        loss_->set_model(this->shared_from_this());
	return *this;
}

template <typename T>
const std::shared_ptr<Loss<T>> WeightedModel<T>::get_loss(void) const
{
	return loss_;
}


template class nn::models::WeightedModel<float>;
// template class nn::models::WeightedModel<double>;



template <typename T>
Perceptron<T>::Perceptron(const Shape &input_shape, const Shape &output_shape, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_shape, output_shape, true, "Perceptron"),
	  dense_(std::make_unique<Dense<T>>(input_shape, output_shape, std::make_shared<StepFunc<T>>(), rand_init))
{
}


template <typename T>
Perceptron<T>::Perceptron(const Shape &input_shape, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_shape, Shape{}, true, "Perceptron"),
	  dense_(std::make_unique<Dense<T>>(input_shape, std::make_shared<StepFunc<T>>(), rand_init))
{
}


template <typename T>
Perceptron<T>::Perceptron(std::size_t input_size, std::size_t output_size, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_size, output_size, true, "Perceptron"),
	  dense_(std::make_unique<Dense<T>>(input_size, output_size, std::make_shared<StepFunc<T>>(), rand_init))
{
}


template <typename T>
Perceptron<T>::Perceptron(std::size_t input_size, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_size, 0, true, "Perceptron"),
	  dense_(std::make_unique<Dense<T>>(input_size, std::make_shared<StepFunc<T>>(), rand_init))
{
}


template <typename T>
Mat<T> &Perceptron<T>::get_weights(void) const
{
	return dense_->get_weights();
}

template <typename T>
Mat<T> &Perceptron<T>::get_bias(void) const
{
	return dense_->get_bias();
}

template <typename T>
Perceptron<T> &Perceptron<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	if (input_shape.rows == 0 || input_shape.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);
	}
	
	if (output_shape.cols == 0 || output_shape.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);
	}

	if (WeightedModel<T>::optimizer_ == nullptr) {
		throw std::invalid_argument("Not seted optimizer");
	}
	
	Layer::input_shape_ = input_shape;
	Layer::output_shape_ = output_shape;

	dense_->build(input_shape, output_shape);
	
	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	dense_->set_optimizer(WeightedLayer::optimizer_);
	
	register_funcs();
	Layer::built_ = true;
	return *this;
}

template <typename T>
Perceptron<T> &Perceptron<T>::build(std::size_t input_size, std::size_t output_size)
{
	if (input_size == 0) {
		throw std::invalid_argument("Invalid input size of the layer: " + Layer::name_);
	}
	
	if (input_size == 0) {
		throw std::invalid_argument("Invalid output size of the layer: " + Layer::name_);
	}

	Layer::input_shape_ = Shape{input_size, 1};
	Layer::output_shape_ = Shape{output_size, 1};

	dense_->build(input_size, output_size);

	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	dense_->set_optimizer(WeightedLayer::optimizer_);
	
	register_funcs();
	Layer::built_ = true;
	return *this;
}

template <typename T>
Perceptron<T> &Perceptron<T>::build(void)
{
	if (Layer::input_shape_.rows == 0 || Layer::input_shape_.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);
	}
	
	if (Layer::output_shape_.cols == 0 || Layer::output_shape_.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);
	}
	
	dense_->build();

	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	dense_->set_optimizer(WeightedLayer::optimizer_);
	
	register_funcs();
	Layer::built_ = true;
	return *this;
}

template <typename T>
Perceptron<T> &Perceptron<T>::fit(const std::shared_ptr<std::vector<Mat<T>>> X_train, const std::shared_ptr<std::vector<Mat<T>>> Y_train, std::size_t nepochs, std::size_t batch_size)
{
	((void) batch_size); // Is not needed 
	
	if (Layer::input_shape_.rows == 0 || Layer::input_shape_.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);
	}
	
	if (Layer::output_shape_.cols == 0 || Layer::output_shape_.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);
	}
	
	if (X_train == nullptr || Y_train == nullptr) {
		throw std::invalid_argument("Inputs and outputs cannot be empty.");
	}
	
	if (X_train->size() != Y_train->size())  {
		throw std::invalid_argument("Inputs and outputs are not of the same size");
	}

	if ((*X_train)[0].get_shape() != Layer::input_shape_) {
		throw std::invalid_argument("Input doesn't match");
	}

	if ((*Y_train)[0].get_shape() != Layer::output_shape_) {
		throw std::invalid_argument("Output doesn't match");
	}

	while (nepochs-- > 0) {
		std::size_t n = X_train->size();
		for (std::size_t i = 0; i < n; i++) {
			Mat<T> Y_pred = (*dense_)((*X_train)[i]);
			if (Y_pred != (*Y_train)[i]) {
				dense_->fit((*Y_train)[i] - Y_pred, (*X_train)[i]);
			}
		}
	}
	
	return *this;
}


template <typename T>
Perceptron<T> &Perceptron<T>::register_funcs(void)
{
	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("feedforward", [this](const Mat<T> &X) -> Mat<T> {
			return (*dense_)(X);
		});

	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("gradient", [this](const Mat<T> &X) -> Mat<T> {
			return dense_->gradient(X);
		});
	
	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("jacobian", [this](const Mat<T> &X) -> Mat<T> {
			return dense_->jacobian(X);
		});

	GenericVTable::register_func<void, const Mat<T> &, const Mat<T> &>
		("fit", [this](const Mat<T> &signal_update, const Mat<T> &input) -> void {
			dense_->fit(signal_update, input);
		});
	
	return *this;
}

template class nn::models::Perceptron<float>;
// template class nn::models::Perceptron<double>;


template <typename T>
Adeline<T>::Adeline(const Shape &input_shape, const Shape &output_shape, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_shape, output_shape, true, "Adeline"),
	  dense_(std::make_unique<Dense<T>>(input_shape, std::make_shared<SigmoidFunc<T>>(), rand_init))
{
}

template <typename T>
Adeline<T>::Adeline(const Shape &input_shape, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_shape, Shape{}, true, "Adeline"),
	  dense_(std::make_unique<Dense<T>>(input_shape, std::make_shared<SigmoidFunc<T>>(), rand_init))
{
}


template <typename T>
Adeline<T>::Adeline(std::size_t input_size, std::size_t output_size, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_size, output_size, true, "Adeline"),
	  dense_(std::make_unique<Dense<T>>(input_size, output_size, std::make_shared<SigmoidFunc<T>>(), rand_init))
{
}

template <typename T>
Adeline<T>::Adeline(std::size_t input_size, std::shared_ptr<RandInitializer> rand_init)
	: WeightedModel<T>(input_size, 0, true, "Adeline"),
	  dense_(std::make_unique<Dense<T>>(input_size, std::make_shared<SigmoidFunc<T>>(), rand_init))
{
}

template <typename T>
Mat<T> &Adeline<T>::get_weights(void) const
{
	return dense_->get_weights();
}

template <typename T>
Mat<T> &Adeline<T>::get_bias(void) const
{
	return dense_->get_bias();
}

template <typename T>
Adeline<T> &Adeline<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	if (input_shape.rows == 0 || input_shape.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);
	}

	if (output_shape.cols == 0 || output_shape.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);
	}

	Layer::input_shape_ = input_shape;
	Layer::output_shape_ = output_shape;

	dense_->build(input_shape, output_shape);

	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	dense_->set_optimizer(WeightedLayer::optimizer_);
	
	register_funcs();
	Layer::built_ = true;
	return *this;
}


template <typename T>
Adeline<T> &Adeline<T>::build(std::size_t input_size, std::size_t output_size)
{
	if (input_size == 0) {
		throw std::invalid_argument("Invalid input size of the layer: " + Layer::name_);
	}
	
	if (input_size == 0) {
		throw std::invalid_argument("Invalid output size of the layer: " + Layer::name_);
	}

	Layer::input_shape_ = Shape{input_size, 1};
	Layer::output_shape_ = Shape{output_size, 1};

	dense_->build(input_size, output_size);

	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	dense_->set_optimizer(WeightedLayer::optimizer_);
	
	register_funcs();
	Layer::built_ = true;
	return *this;
}


template <typename T>
Adeline<T> &Adeline<T>::build(void)
{
	if (Layer::input_shape_.rows == 0 || Layer::input_shape_.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);
	}
	
	if (Layer::output_shape_.cols == 0 || Layer::output_shape_.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);
	}
	
	dense_->build();

	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	dense_->set_optimizer(WeightedLayer::optimizer_);
	
	register_funcs();
	Layer::built_ = true;
	return *this;
}


template <typename T>
Adeline<T> &Adeline<T>::fit(const std::shared_ptr<std::vector<Mat<T>>> X_train, const std::shared_ptr<std::vector<Mat<T>>> Y_train, std::size_t nepochs, std::size_t batch_size)
{
	((void) batch_size); // Is not needed
	
	if (Layer::input_shape_.rows == 0 || Layer::input_shape_.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);
	}
	
	if (Layer::output_shape_.cols == 0 || Layer::output_shape_.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);
	}
	
	if (X_train == nullptr || Y_train == nullptr) {
		throw std::invalid_argument("Inputs and outputs cannot be empty.");
	}
	
	if (X_train->size() != Y_train->size())  {
		throw std::invalid_argument("Inputs and outputs are not of the same size");
	}

	if ((*X_train)[0].get_shape() != Layer::input_shape_) {
		throw std::invalid_argument("Input doesn't match");
	}

	if ((*Y_train)[0].get_shape() != Layer::output_shape_) {
		throw std::invalid_argument("Output doesn't match");
	}

	while (nepochs-- > 0) {
		std::size_t n = X_train->size();
		for (std::size_t i = 0; i < n; i++) {
			// Compute the gradient,
			Mat<T> Z = this->dense_->get_weights().dot((*X_train)[i]) + this->dense_->get_bias();
			// dY/dZ = Y * (1 - Y), since Y = s(Z), where s(Z) = 1 / (1 - e^{- Z})
			// grad_Y_Z in R^{m, 1}
			Mat<T> grad_Y_Z = this->dense_->get_activation_func()->gradient(Z);
			// dL/dY = (Y - T) / (Y * (1 - Y)), where T is the true label
			// grad_L_Y in R^{m, 1}
			Mat<T> grad_L_Y = this->get_loss()->gradient({(*X_train)[i], (*Y_train)[i]});
			
			// Element-Wise product
			Mat<T> grad_L_Z = grad_L_Y * grad_Y_Z;
			this->dense_->fit(grad_L_Z, (*X_train)[i]);
		}
	}
	
	return *this;
}

template <typename T>
Adeline<T> &Adeline<T>::register_funcs(void)
{
	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("feedforward", [this](const Mat<T> &X) -> Mat<T> {
			return (*dense_)(X);
		});

	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("gradient", [this](const Mat<T> &X) -> Mat<T> {
			return dense_->gradient(X);
		});

	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("jacobian", [this](const Mat<T> &X) -> Mat<T> {
			return dense_->jacobian(X);
		});


	GenericVTable::register_func<void, const Mat<T> &, const Mat<T> &>
		("fit", [this](const Mat<T> &signal_update, const Mat<T> &input) -> void {
			dense_->fit(signal_update, input);
		});

	return *this;
}



template class nn::models::Adeline<float>;
// template class nn::models::Adeline<double>;




template <typename T>
Sequential<T>::Sequential(std::initializer_list<std::unique_ptr<Layer>> layers_init)
{
    for (auto& l : layers_init)
        layers_.push_back(std::move(const_cast<std::unique_ptr<Layer>&>(l)));
}

// TODO: Add some validations here we need 

template <typename T>
Sequential<T> &Sequential<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	((void) input_shape);
	((void) output_shape);

	
	return this->build();
}


template <typename T>
Sequential<T> &Sequential<T>::build(std::size_t input_size, std::size_t output_size)
{
	((void) input_size);
	((void) output_size);
	
	return this->build();
}

template <typename T>
Sequential<T> &Sequential<T>::build(void)
{
	if (WeightedLayer::optimizer_ == nullptr) {
		throw std::invalid_argument("Not set an optimizer");
	}

	for (auto &layer_ptr : layers_) {
		layer_ptr->build();
		if (layer_ptr->is_trainable()) {
			((WeightedLayer *) layer_ptr.get())->set_optimizer(WeightedLayer::optimizer_);
		}
	}

	this->set_input_shape(layers_[0]->get_input_shape());
	this->set_output_shape(layers_.back()->get_output_shape());
	
	this->register_funcs();

	
	return *this;
}


template <typename T>
Sequential<T> &Sequential<T>::fit(const std::shared_ptr<std::vector<Mat<T>>> X_train,
                                  const std::shared_ptr<std::vector<Mat<T>>> Y_train,
                                  std::size_t nepochs, std::size_t batch_size)
{
    ((void) batch_size);

    if (Layer::input_shape_.rows == 0 || Layer::input_shape_.cols == 0)
        throw std::invalid_argument("Invalid input shape of the layer: " + Layer::name_);

    if (Layer::output_shape_.cols == 0 || Layer::output_shape_.cols == 0)
        throw std::invalid_argument("Invalid output shape of the layer: " + Layer::name_);

    if (!X_train || !Y_train)
        throw std::invalid_argument("Inputs and outputs cannot be empty.");

    if (X_train->size() != Y_train->size())
        throw std::invalid_argument("Inputs and outputs are not of the same size");

    if ((*X_train)[0].get_shape() != Layer::input_shape_)
        throw std::invalid_argument("Input doesn't match");

    if ((*Y_train)[0].get_shape() != Layer::output_shape_)
        throw std::invalid_argument("Output doesn't match");

    // Prepare the model
    this->loss_->set_inputs(X_train);
    this->loss_->set_outputs(Y_train);
    this->loss_->set_model(this->shared_from_this());

    while (nepochs-- > 0) {
        std::size_t n = X_train->size();

        for (std::size_t i = 0; i < n; i++) {
            const Mat<T> &x = (*X_train)[i];


            // Hardcodeamos la parte de CrossEntropy + Sigmoid
	    // simplificado: dL/dz = (a - y)
	    // const Mat<T> &y_true = (*Y_train)[i];
            // Mat<T> y_pred = (*this)(x);
            // Mat<T> grad(y_pred.get_shape());
            // for (std::size_t r = 0; r < y_pred.rows(); ++r)
	    // 	    for (std::size_t c = 0; c < y_pred.cols(); ++c)
	    // 		    grad(r, c) = y_pred(r, c) - y_true(r, c);

	    Mat<T> grad = this->loss_->gradient(std::make_pair((*X_train)[i], (*Y_train)[i]));
	    // std::cout << "epoch: " << nepochs << " grad: " << grad(0, 0) << std::endl;
            this->WeightedLayer::fit(grad, x);
        }
    }

    return *this;
}




template <typename T>
Sequential<T> &Sequential<T>::register_funcs(void)
{
	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("feedforward", [this](const Mat<T> &X) -> Mat<T> {
			Mat<T> A_prev = X;
			Mat<T> A_next;
			for (auto &layer_ptr : this->layers_) {
				A_next = (*layer_ptr)(A_prev);
				A_prev = A_next;
			}

			return A_next;
		});


	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("gradient", [this](const Mat<T> &X) -> Mat<T> {
			// Not implemented yet
			return Mat<T>(1, 1).fill(1);
		});
	


	GenericVTable::register_func<Mat<T>, const Mat<T> &>
		("jacobian", [this](const Mat<T> &X) -> Mat<T> {
			// Not implemented yet
			return Mat<T>(1, 1).fill(1);
		});


	GenericVTable::register_func<void, const Mat<T> &, const Mat<T> &>
		("fit", [this](const Mat<T> &dE_dY, const Mat<T> &X) -> void {
			std::vector<Mat<T>> inputs;
			inputs.push_back(X);

			// Get each input from the layers
			for (auto &layer_ptr : this->layers_) {
				inputs.push_back((*layer_ptr)(inputs.back()));
			}
			// Remove the output from the network
			inputs.pop_back();

			// Iterate by reverse
			// Here we are supposing that the `dE_dY` in R^{m, 1}
			Mat<T> dL_dA_prev = dE_dY;
			for (int i = layers_.size() - 1; i >= 0; i--) {
				if (layers_[i].get()->is_trainable()) {
					// Commeted out because we are already getting the epxression of a - y from above
					if (static_cast<WeightedLayer *>(layers_[i].get())->has_activation_func()
					    // && i < layers_.size() - 1
					    ) {
						// (m, n) . (n, 1) + (m, 1) = (m, 1)
						Mat<T> Z = static_cast<Dense<T> *>(layers_[i].get())->get_weights().dot(inputs[i])
							+ static_cast<Dense<T> *>(layers_[i].get())->get_bias();
						// (m, 1) . (m, 1) = (m, 1)
						Mat<T> dL_dZ = static_cast<WeightedLayer *>(layers_[i].get())->get_activation_func()->gradient(Z) * dL_dA_prev;
						// std::cout << "dL_dZ: " << dL_dZ << std::endl;
						static_cast<WeightedLayer *>(layers_[i].get())->fit(dL_dZ, inputs[i]);
					} else {
						static_cast<WeightedLayer *>(layers_[i].get())->fit(dL_dA_prev, inputs[i]);
					}
				}
				Mat<T> dL_dA = layers_[i].get()->jacobian(inputs[i]).dot(dL_dA_prev);
				dL_dA_prev = dL_dA;
			}
		});
	
	return *this;
}


template <typename T>
const std::vector<std::unique_ptr<Layer>> &nn::models::Sequential<T>::get_layers(void) const
{
	return layers_;
}

template class nn::models::Sequential<float>;



