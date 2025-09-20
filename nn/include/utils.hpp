#ifndef NN_UTILS_INCLUDED
#define NN_UTILS_INCLUDED

#include <any>
#include <functional>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <vector>
#include <unordered_map>

namespace nn::utils {

	/**
	 * @class GenericVTable
	 * @brief A generic runtime "virtual table" that allows registering and retrieving
	 *        functions by name and signature, effectively simulating virtual template methods.
	 *
	 * The purpose of this class is to provide a way to declare template-like virtual methods
	 * inside classes. Derived classes can register implementations of functions with arbitrary
	 * signatures, and later retrieve them dynamically at runtime.
	 *
	 * ### Key ideas:
	 * - Each function is identified by a name + type signature (return + argument types).
	 * - Signatures are internally stored as `(std::string, std::vector<std::type_index>)`.
	 * - Functions are wrapped into `std::function` and stored in a type-erased `std::any`.
	 * - If a requested function is missing, a detailed `std::runtime_error` is thrown.
	 *
	 * Example:
	 * @code
	 * class MyLayer : public GenericVTable {
	 * public:
	 *     MyLayer() {
	 *         register_func<int, int>("double", [](int x) { return x * 2; });
	 *     }
	 *
	 *     int apply(int x) {
	 *         return get_func<int, int>("double")(x);
	 *     }
	 * };
	 * @endcode
	 */
	class GenericVTable {
	public:
		virtual ~GenericVTable(void) = 0;
		
		/**
		 * @brief Register a function under a given name and type signature.
		 *
		 * @tparam Ret   Return type of the function.
		 * @tparam Args  Parameter types of the function.
		 * @tparam F     Callable type (lambda, function pointer, etc).
		 *
		 * @param name   Symbolic name for the function (e.g., "forward", "sum").
		 * @param func   Callable object implementing the function.
		 *
		 * Example:
		 * @code
		 * register_func<int, int>("square", [](int x){ return x * x; });
		 * @endcode
		 */
		template<typename Ret, typename... Args, typename F>
		void register_func(const std::string& name, F&& func)
		{
			auto key = make_signature<Ret, Args...>(name);
			vtable_[key] = std::function<Ret(Args...)>(std::forward<F>(func));
		}

		/**
		 * @brief Retrieve a previously registered function by name and type signature.
		 *
		 * @tparam Ret   Expected return type.
		 * @tparam Args  Expected parameter types.
		 *
		 * @param name   Name of the function to look up.
		 * @param file   Optional file name for debugging (defaults to nullptr).
		 * @param line   Optional line number for debugging (defaults to -1).
		 *
		 * @throws std::runtime_error if the function is not found or signature mismatch.
		 *
		 * Example:
		 * @code
		 * auto f = get_func<int, int>("square");
		 * int result = f(4); // -> 16
		 * @endcode
		 */
		template<typename Ret, typename... Args>
		std::function<Ret(Args...)> get_func(const std::string& name,
		                                     const char* file = nullptr,
		                                     int line = -1) const 
		{
			auto key = make_signature<Ret, Args...>(name);
			auto it = vtable_.find(key);
			if (it == vtable_.end()) {
				std::string msg = "Function not implemented: " + name;
				if (file) msg += " at " + std::string(file) + ":" + std::to_string(line);
				throw std::runtime_error(msg);
			}
			return std::any_cast<std::function<Ret(Args...)>>(it->second);
		}

	protected:
		/**
		 * @brief Registers the functions required by the derived class.
		 *
		 * This virtual method is intended to be overridden by subclasses to
		 * populate and customize their function table (vtable-like structure).
		 * 
		 * @return A reference to the updated GenericVTable containing the 
		 *         registered functions.
		 *
		 * @note 
		 * - The base implementation may provide common registrations.
		 * - Derived classes should extend or override this to register their
		 *   specific functionality.
		 */
		virtual GenericVTable &register_funcs(void) = 0;
		

	private:
		/// Composite key: function name + type signature
		using FuncKey = std::pair<std::string, std::vector<std::type_index>>;

		/// Utility: create a function signature key
		template<typename Ret, typename... Args>
		static FuncKey make_signature(const std::string& name)
		{
			return { name, { typeid(Ret), typeid(Args)... } };
		}

		/// Custom hasher for FuncKey
		struct FuncKeyHash {
			std::size_t operator()(const FuncKey& key) const {
				std::size_t h = std::hash<std::string>{}(key.first);
				for (auto& t : key.second) {
					h ^= t.hash_code() + 0x9e3779b9 + (h << 6) + (h >> 2);
				}
				return h;
			}
		};

		/// Custom equality comparator for FuncKey
		struct FuncKeyEqual {
			bool operator()(const FuncKey& a, const FuncKey& b) const {
				if (a.first != b.first || a.second.size() != b.second.size())
					return false;
				for (size_t i = 0; i < a.second.size(); ++i)
					if (a.second[i] != b.second[i]) return false;
				return true;
			}
		};

		/// The actual vtable storage (maps signature -> function)
		std::unordered_map<FuncKey, std::any, FuncKeyHash, FuncKeyEqual> vtable_;
	};
} 

#endif
