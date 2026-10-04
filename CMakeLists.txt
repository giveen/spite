cmake_minimum_required(VERSION 3.25)
project(spite_kernels LANGUAGES C CXX CUDA)

# C++23 everywhere
set(CMAKE_C_STANDARD            17)
set(CMAKE_CXX_STANDARD          23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         20)   # CUDA 12.x supports C++20; 23 coming
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── User build selections ─────────────────────────────────────────────────
# Model specs are "<family>/<model>" pairs matching the kernels/ tree:
# cmake -DSPITE_MODELS="llama/llama3,mistral/mistral4" -DSPITE_GPU_ARCHS="sm_89,rdna3"
set(SPITE_MODELS      "llama/llama3"  CACHE STRING "Comma-separated <family>/<model> list")
set(SPITE_GPU_ARCHS   "sm_89"         CACHE STRING "Comma-separated GPU arch list")
set(SPITE_QUANT_TYPES "q4_k,q8_0"     CACHE STRING "Comma-separated quant types (informational)")

string(REPLACE "," ";" _model_specs ${SPITE_MODELS})
string(REPLACE "," ";" _archs       ${SPITE_GPU_ARCHS})

message(STATUS "spite kernels: models=${_model_specs}  archs=${_archs}")

# ── CUDA arch flags ───────────────────────────────────────────────────────
set(_cuda_archs "")
foreach(_a ${_archs})
    if(_a MATCHES "^sm_([0-9]+)$")
        list(APPEND _cuda_archs ${CMAKE_MATCH_1})
    endif()
endforeach()
if(_cuda_archs)
    set(CMAKE_CUDA_ARCHITECTURES ${_cuda_archs})
    message(STATUS "spite kernels: CUDA architectures=${_cuda_archs}")
endif()

# ── Common CUDA flags ─────────────────────────────────────────────────────
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)

# ── Kernel targets ────────────────────────────────────────────────────────
# For each (<family>/<model>, gpu_arch) pair, compile all .cu/.cpp/.c files
# into a shared library: libkernel_<family>_<model>_<arch>.so
foreach(_spec ${_model_specs})
    string(REPLACE "/" ";" _parts "${_spec}")
    list(LENGTH _parts _n_parts)
    if(NOT _n_parts EQUAL 2)
        message(WARNING "spite kernels: skipping '${_spec}' — expected <family>/<model>")
        continue()
    endif()
    list(GET _parts 0 _family)
    list(GET _parts 1 _model)

    foreach(_arch ${_archs})
        set(_dir "${CMAKE_SOURCE_DIR}/kernels/${_family}/${_model}/${_arch}")
        if(NOT IS_DIRECTORY "${_dir}")
            continue()
        endif()

        file(GLOB _srcs
            "${_dir}/*.cu"
            "${_dir}/*.cpp"
            "${_dir}/*.c"
        )
        list(FILTER _srcs EXCLUDE REGEX "KERNEL_TEMPLATE")

        if(NOT _srcs)
            continue()
        endif()

        set(_target "kernel_${_family}_${_model}_${_arch}")
        add_library(${_target} SHARED ${_srcs})
        target_include_directories(${_target} PRIVATE "${CMAKE_SOURCE_DIR}")

        # HIP kernels: swap CUDA for HIP
        if(_arch MATCHES "^rdna|^rx")
            find_package(hip QUIET)
            if(hip_FOUND)
                target_link_libraries(${_target} PRIVATE hip::host)
            endif()
        endif()

        set_target_properties(${_target} PROPERTIES
            OUTPUT_NAME      "kernel_${_family}_${_model}_${_arch}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/${_family}/${_model}/${_arch}"
        )

        message(STATUS "spite kernels: + ${_family}/${_model}/${_arch}")
    endforeach()
endforeach()

# ── Generic fallback (always built, CPU only) ─────────────────────────────
file(GLOB _generic_srcs "kernels/generic/generic/*.c" "kernels/generic/generic/*.cpp")
if(_generic_srcs)
    add_library(kernel_generic SHARED ${_generic_srcs})
    target_include_directories(kernel_generic PRIVATE "${CMAKE_SOURCE_DIR}")
    set_target_properties(kernel_generic PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/generic/generic"
    )
endif()
