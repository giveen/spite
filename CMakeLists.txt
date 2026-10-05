cmake_minimum_required(VERSION 3.25)
project(spite_kernels LANGUAGES C CXX CUDA)

# C++23 everywhere
set(CMAKE_C_STANDARD            17)
set(CMAKE_CXX_STANDARD          23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         20)   # CUDA 12.x supports C++20; 23 coming
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── User build selections ─────────────────────────────────────────────────
# Model specs are "<family>/<model>" pairs matching the kernels/ tree.
# SPITE_GPU_ARCHS accepts card names (RTX_4090, RX_7800_XT, Arc_A770, M4_Pro)
# or raw arch strings (sm_89, rdna3, arc_alchemist, metal) — both work.
# cmake -DSPITE_MODELS="llama/llama4" -DSPITE_GPU_ARCHS="RTX_4090"
set(SPITE_MODELS      "llama/llama3"  CACHE STRING "Comma-separated <family>/<model> list")
set(SPITE_GPU_ARCHS   "RTX_4090"      CACHE STRING "Comma-separated GPU card or arch list")
set(SPITE_QUANT_TYPES "q4_k,q8_0"     CACHE STRING "Comma-separated quant types (informational)")

string(REPLACE "," ";" _model_specs ${SPITE_MODELS})
string(REPLACE "," ";" _archs       ${SPITE_GPU_ARCHS})

# ── Card-to-arch mapping ──────────────────────────────────────────────────
# Translates a card name to a GPU arch string; passes arch strings through.
function(spite_card_to_arch _in _out)
    string(TOUPPER "${_in}" _u)
    string(REPLACE " " "_" _u "${_u}")
    if    (_u MATCHES "^(RTX_5090|RTX_5080|RTX_5070_TI|RTX_5070|RTX_5060_TI|RTX_5060)$")
        set(${_out} "sm_120" PARENT_SCOPE)
    elseif(_u MATCHES "^(RTX_4090|RTX_4080_SUPER|RTX_4080|RTX_4070_TI_SUPER|RTX_4070_TI|RTX_4070_SUPER|RTX_4070|RTX_4060_TI|RTX_4060|RTX_4000_ADA|RTX_5000_ADA|RTX_6000_ADA)$")
        set(${_out} "sm_89" PARENT_SCOPE)
    elseif(_u MATCHES "^(RTX_3090_TI|RTX_3090|RTX_3080_TI|RTX_3080|RTX_3070_TI|RTX_3070|RTX_3060_TI|RTX_3060|RTX_A2000|RTX_A4000|RTX_A4500|RTX_A5000|RTX_A6000)$")
        set(${_out} "sm_86" PARENT_SCOPE)
    elseif(_u MATCHES "^(RTX_2080_TI|RTX_2080_SUPER|RTX_2080|RTX_2070_SUPER|RTX_2070|RTX_2060_SUPER|RTX_2060|GTX_1660_TI|GTX_1660_SUPER|GTX_1660)$")
        set(${_out} "sm_75" PARENT_SCOPE)
    elseif(_u MATCHES "^(RX_9070_XT|RX_9070|RX_9070_GRE|RX_9060_XT|RX_9060)$")
        set(${_out} "rdna4" PARENT_SCOPE)
    elseif(_u MATCHES "^(RX_7900_XTX|RX_7900_XT|RX_7900_GRE|RX_7800_XT|RX_7700_XT|RX_7600_XT|RX_7600)$")
        set(${_out} "rdna3" PARENT_SCOPE)
    elseif(_u MATCHES "^(RX_6950_XT|RX_6900_XT|RX_6800_XT|RX_6800|RX_6750_XT|RX_6700_XT|RX_6650_XT|RX_6600_XT|RX_6600)$")
        set(${_out} "rdna2" PARENT_SCOPE)
    elseif(_u MATCHES "^(RX_5700_XT|RX_5700|RX_5600_XT|RX_5500_XT)$")
        set(${_out} "rdna1" PARENT_SCOPE)
    elseif(_u MATCHES "^(ARC_B580|ARC_B570)$")
        set(${_out} "arc_battlemage" PARENT_SCOPE)
    elseif(_u MATCHES "^(ARC_A770|ARC_A750|ARC_A580|ARC_A380|ARC_A310)$")
        set(${_out} "arc_alchemist" PARENT_SCOPE)
    elseif(_u MATCHES "^(M1|M1_PRO|M1_MAX|M1_ULTRA|M2|M2_PRO|M2_MAX|M2_ULTRA|M3|M3_PRO|M3_MAX|M4|M4_PRO|M4_MAX)$")
        set(${_out} "metal" PARENT_SCOPE)
    else()
        set(${_out} "${_in}" PARENT_SCOPE)
    endif()
endfunction()

set(_resolved_archs "")
foreach(_a ${_archs})
    spite_card_to_arch("${_a}" _r)
    list(APPEND _resolved_archs "${_r}")
endforeach()
set(_archs ${_resolved_archs})

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

        # CUDA kernels share the host's libcudart (dlopen'd by spite-gpu) so
        # device pointers and the primary context are one runtime instance.
        if(_arch MATCHES "^sm_")
            find_package(CUDAToolkit REQUIRED)
            set_target_properties(${_target} PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
                INSTALL_RPATH        "${CUDAToolkit_LIBRARY_DIR}"
            )
        endif()

        set_target_properties(${_target} PROPERTIES
            OUTPUT_NAME      "kernel_${_family}_${_model}_${_arch}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/kernels/${_family}/${_model}/${_arch}"
        )

        # `cmake --install build --prefix .` copies the .so into ./kernels/,
        # which the host checks by default.
        install(TARGETS ${_target}
            LIBRARY DESTINATION kernels/${_family}/${_model}/${_arch}
            RUNTIME DESTINATION kernels/${_family}/${_model}/${_arch}
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
    install(TARGETS kernel_generic
        LIBRARY DESTINATION kernels/generic/generic
        RUNTIME DESTINATION kernels/generic/generic
    )
endif()
