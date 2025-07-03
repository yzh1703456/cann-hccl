set_source_files_properties(
    aiv_communication.h PROPERTIES LANGUAGE CXX
)

if(BUILD_OPEN_PROJECT)
    set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
    add_custom_target(hccl_aiv_op_ascend910B ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o --cce-aicore-arch=dav-c220-vec
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface/
            -I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl/
            -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
else()
    get_filename_component(CCE_DIR "${CMAKE_LINKER}" PATH)

    set(tiling_script ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/cmake/scripts/gen_kernel_tiling_data_def.py)
    set(TILING_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(tiling_gen_file ${TILING_GEN_DIR}/kernel_tiling/kernel_tiling.h)
    set(tiling_data_def_path ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/lib)

    add_custom_command(OUTPUT ${tiling_gen_file}
        COMMAND python3 ${tiling_script} ${tiling_data_def_path} ${tiling_gen_file}
        DEPENDS ${tiling_script} ${tiling_data_def_path})

    add_custom_target(hccl_aiv_op_ascend910B
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o --cce-aicore-arch=dav-c220-vec
            --cce-aicore-input-parameter-size=28000 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/interface/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcfw/impl/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/inc/
            -I${TOP_DIR}/atc/opcompiler/ascendc_compiler/framework/tikcpulib/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${tiling_gen_file}
    )
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_ascend910B.o OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)