set(HCCL_PACKET_FLAG 1)

if(BUILD_OPEN_PROJECT)
    set(HCCL_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
else()
    set(HCCL_BASE_DIR ${TOP_DIR}/ace/comop/hccl/whole/hccl/)
endif()

add_library(hccl  SHARED)

set(hccl_include_list
    ${HCCL_BASE_DIR}/algorithm/base/inc
    ${HCCL_BASE_DIR}/algorithm/pub_inc
    ${HCCL_BASE_DIR}/algorithm/alg_research/alg_template
    ${HCCL_BASE_DIR}/pub_inc
    ${HCCL_BASE_DIR}/pub_inc/hccl
    ${HCCL_BASE_DIR}/algorithm/base/alg_aiv_template
    ${HCCL_BASE_DIR}/algorithm/base/alg_template
    ${HCCL_BASE_DIR}/algorithm/base/mc2_handler
    ${HCCL_BASE_DIR}/algorithm/base/alg_template/component/inc
    ${HCCL_BASE_DIR}/algorithm/base/communicator
    ${HCCL_BASE_DIR}/algorithm/base/communicator/legacy
    ${HCCL_BASE_DIR}/algorithm/impl
    ${HCCL_BASE_DIR}/algorithm/impl/legacy
    ${HCCL_BASE_DIR}/algorithm/impl/inc
    ${HCCL_BASE_DIR}/algorithm/impl/task
    ${HCCL_BASE_DIR}/algorithm/impl/resource_manager
    ${HCCL_BASE_DIR}/algorithm/impl/coll_executor
    ${HCCL_BASE_DIR}/common/debug/profiling/inc

    ${HCCL_BASE_DIR}/framework/cluster_maintenance/health/heartbeat/
    ${HCCL_BASE_DIR}/framework/cluster_maintenance/recovery/operator_retry/
    ${HCCL_BASE_DIR}/framework/common/src/
    ${HCCL_BASE_DIR}/framework/common/src/config
    ${HCCL_BASE_DIR}/framework/common/src/topo
    ${HCCL_BASE_DIR}/framework/common/src/task
    ${HCCL_BASE_DIR}/framework/common/src/exception
    ${HCCL_BASE_DIR}/framework/common/src/thread
    ${HCCL_BASE_DIR}/framework/communicator/impl/
    ${HCCL_BASE_DIR}/framework/communicator/impl/resource_manager
    ${HCCL_BASE_DIR}/framework/communicator/impl/one_sided_service
    ${HCCL_BASE_DIR}/framework/communicator/impl/zero_copy_memory_agent
    ${HCCL_BASE_DIR}/framework/hcom/
    ${HCCL_BASE_DIR}/framework/inc/
    ${HCCL_BASE_DIR}/framework/op_base/src
    ${HCCL_BASE_DIR}/common/health/
    ${HCCL_BASE_DIR}/common/stream/
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP)

    target_include_directories(hccl PRIVATE
        ${HCCL_BASE_DIR}/../../../inc
        ${HCCL_BASE_DIR}/../../../inc/hccl
        ${HCCL_BASE_DIR}/../../../hccl/
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
        ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api/
        ${THIRD_PARTY_NLOHMANN_PATH}
    )

    target_link_directories(hccl PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
else()
    target_include_directories(hccl PRIVATE
        ${TOP_DIR}/ace/comop/inc/external/hccl/             # ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TOP_DIR}/ace/comop/inc/hccl/                      # ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TOP_DIR}/ace/comop/hccl/whole/hccl/algorithm/alg_research/pub_inc
        ${TOP_DIR}/abl/msprof/inc                           # experiment/msprof/
        ${TOP_DIR}/ace/npuruntime/inc                       # experiment/runtime/
        ${TOP_DIR}/inc/driver                               # experiment/ascend_hal/driver/
        ${TOP_DIR}/inc/external                             # experiment
        ${TOP_DIR}/metadef/inc/common/util/error_manager    # experiment/metadef/inc/common/util/error_manager/
        ${TOP_DIR}/metadef/inc/external/
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/abl/slog/inc/toolchain                   # experiment/slog/
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/mmpa/inc/mmpa
        ${TOP_DIR}/ace/comop/hccl/
    )
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_options(hccl PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++14
    -fstack-protector-all
)

target_link_options(hccl PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_compile_definitions(hccl PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
)
target_include_directories(hccl PRIVATE
    ${hccl_include_list}
)

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl
            -Wl,--no-as-needed
            c_sec
            ascendalog
            hccl_alg
            hccl_plf
            -Wl,--as-needed
            -lrt
            -ldl
            -lpthread
    )
else()
    target_link_libraries(hccl
            $<BUILD_INTERFACE:kernel_tiling_headers>
            $<BUILD_INTERFACE:slog_headers>
            $<BUILD_INTERFACE:mmpa_headers>
            -Wl,--no-as-needed
            c_sec
            slog
            mmpa
            hccl_alg
            hccl_plf
            hccl_v2
            -Wl,--as-needed
            -lrt
            -ldl
            -lpthread
    )
endif()

add_subdirectory(common)
add_subdirectory(communicator)
add_subdirectory(hcom)
add_subdirectory(op_base)
add_subdirectory(cluster_maintenance)

install(TARGETS hccl
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)
