cmake_minimum_required(VERSION 3.20)
project(spite C CXX CUDA)

set(CMAKE_C_STANDARD   17)
set(CMAKE_CXX_STANDARD 17)

# ── User build selections ────────────────────────────────────────────────
# Comma-separated lists. Only the selected combos are compiled.
#   cmake -DSPITE_MODELS="llama3,mistral" -DSPITE_GPU_ARCHS="sm_89,sm_86"
set(SPITE_MODELS     "llama3"   CACHE STRING "Comma-separated model list")
set(SPITE_GPU_ARCHS  "sm_89"    CACHE STRING "Comma-separated GPU arch list")
set(SPITE_QUANT_TYPES "q4_k,q8_0" CACHE STRING "Comma-separated quant types")

string(REPLACE "," ";" SPITE_MODELS_LIST     ${SPITE_MODELS})
string(REPLACE "," ";" SPITE_GPU_ARCHS_LIST  ${SPITE_GPU_ARCHS})
string(REPLACE "," ";" SPITE_QUANT_LIST      ${SPITE_QUANT_TYPES})

message(STATUS "spite: models=${SPITE_MODELS_LIST}")
message(STATUS "spite: gpu_archs=${SPITE_GPU_ARCHS_LIST}")
message(STATUS "spite: quant_types=${SPITE_QUANT_LIST}")

# ── CUDA arch flags ──────────────────────────────────────────────────────
set(SPITE_CUDA_ARCHS "")
foreach(arch ${SPITE_GPU_ARCHS_LIST})
    if(arch MATCHES "^sm_")
        string(REPLACE "sm_" "" num ${arch})
        list(APPEND SPITE_CUDA_ARCHS ${num})
    endif()
endforeach()
if(SPITE_CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${SPITE_CUDA_ARCHS})
endif()

# ── Core library ─────────────────────────────────────────────────────────
add_library(spite_core STATIC
    core/quant.c
    loader/gguf.c
    dispatch/dispatch.c
)
target_include_directories(spite_core PUBLIC ${CMAKE_SOURCE_DIR})

# ── Kernel targets ───────────────────────────────────────────────────────
# For each (model, gpu_arch) pair, glob the source files and compile them
# into a shared library: kernels/spite_kernel_<model>_<arch>.so
foreach(model ${SPITE_MODELS_LIST})
    foreach(arch ${SPITE_GPU_ARCHS_LIST})
        set(kernel_dir "${CMAKE_SOURCE_DIR}/kernels/${model}/${arch}")
        if(IS_DIRECTORY ${kernel_dir})
            file(GLOB kernel_srcs
                "${kernel_dir}/*.cu"
                "${kernel_dir}/*.c"
                "${kernel_dir}/*.cpp"
                "${kernel_dir}/*.cl"
            )
            # Skip template files
            list(FILTER kernel_srcs EXCLUDE REGEX "KERNEL_TEMPLATE")

            if(kernel_srcs)
                set(target_name "kernel_${model}_${arch}")
                add_library(${target_name} SHARED ${kernel_srcs})
                target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR})
                set_target_properties(${target_name} PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels"
                )
                message(STATUS "spite: building kernel ${model}/${arch}")
            endif()
        endif()
    endforeach()
endforeach()

# ── Generic fallback kernel (always built) ───────────────────────────────
add_library(kernel_generic SHARED
    kernels/generic/generic/ops.c
)
target_include_directories(kernel_generic PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(kernel_generic PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels"
)

# ── CLI executable ───────────────────────────────────────────────────────
add_executable(spite
    tools/main.c
)
target_link_libraries(spite spite_core dl)
