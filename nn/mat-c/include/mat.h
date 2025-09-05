#ifndef MAT_INCLUDED
#define MAT_INCLUDED

#include <stddef.h>

/* memory_allocator_func_type: function pointer type for allocating memory */
typedef void *(*memory_allocator_func_type)(size_t);

/* memory_free_func_type: function pointer type for freeing memory */
typedef void (memory_free_func_type)(void *);

/* MatLibAllocatorType: struct holding function pointers for memory allocation and freeing */
typedef struct MatLibAllocatorType {
	memory_allocator_func_type mem_alloc; /* allocator function */
	memory_free_func_type mem_free;       /* free function */
} MatLibAllocatorType;

/* current_mat_lib_allocator: pointer to the currently used allocator */
extern MatLibAllocatorType *current_mat_lib_allocator;


#endif
