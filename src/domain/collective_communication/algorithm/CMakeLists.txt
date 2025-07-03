cmake_minimum_required(VERSION 3.16.0)
project(hccl_alg)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_library(hccl_alg SHARED)

set(HCCL_BASE_DIR ${TOP_DIR}/ace/comop/hccl/whole/hccl/)

target_include_directories(hccl_alg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/base/inc
    ${HCCL_BASE_DIR}/pub_inc
    ${HCCL_BASE_DIR}/pub_inc/hccl
    ${HCCL_BASE_DIR}/algorithm/alg_research/alg_template
    ${HCCL_BASE_DIR}/inc
    ${HCCL_BASE_DIR}/inc/adapter
    ${TOP_DIR}/inc
    ${TOP_DIR}/open_source/json/include
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/ace/npuruntime/acl/inc/external
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_aiv_template
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/component/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/base/mc2_handler
    ${CMAKE_CURRENT_SOURCE_DIR}/base/communicator
    ${CMAKE_CURRENT_SOURCE_DIR}/base/communicator/legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/legacy/operator
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/resource_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/task
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/operator
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/operator/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_send_receive
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_reduce/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_to_all
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather_v
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather_v/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter_v
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter_v/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_broadcast/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/pub_inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/health
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/debug/profiling/inc
    ${HCCL_BASE_DIR}/../../../../inc/external
    ${HCCL_BASE_DIR}/../../../../inc/external/hccl
    ${HCCL_BASE_DIR}/../../../../inc/hccl
    ${HCCL_BASE_DIR}/../../../../inc
    ${TOP_DIR}/ace/comop/inc/external/hccl
    ${TOP_DIR}/ace/comop/inc/hccl
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/algorithm/alg_research/pub_inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/inc/adapter/
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/inc
    ${TOP_DIR}/drivers/network/inc
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl_alg PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP)

    target_include_directories(hccl_alg PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../pub_inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
    )

    target_link_directories(hccl_alg PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
else()
    target_include_directories(hccl_alg PRIVATE
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/inc
        ${TOP_DIR}/metadef/inc/common/util/error_manager/
        ${TOP_DIR}/metadef/inc/external/
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/mmpa/inc/mmpa
    )
endif()

target_compile_definitions(hccl_alg PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
)

target_compile_options(hccl_alg PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++14
    -fstack-protector-all
)

target_link_options(hccl_alg PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl_alg PRIVATE
        -Wl,--no-as-needed
        c_sec
        ascendalog
        -Wl,--no-as-needed
        hccl_alg_research
        hccl_plf
    )
else()
    target_link_libraries(hccl_alg PRIVATE
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:runtime_headers> #runtime/rt_error_codes.h
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        mmpa
        -Wl,--no-as-needed
        hccl_alg_research
        hccl_plf
        ofed_headers
    )
endif()

add_subdirectory(base)
add_subdirectory(impl)

install(TARGETS hccl_alg
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

if(NOT BUILD_OPEN_PROJECT)
    install_package(
        PACKAGE hccl
        TARGETS hccl_alg
    )
endif()

if(BUILD_OPEN_PROJECT)
    set(install_script_dir ${CMAKE_CURRENT_BINARY_DIR}/install_scripts/)

    add_custom_target(generate_install_script ALL
        COMMAND rm -rf ${install_script_dir}
        COMMAND cp -rf ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/open_install_scripts ${install_script_dir}
        COMMAND chmod -R u+w ${install_script_dir}
        COMMAND echo "base_package=hccl" > ${install_script_dir}/version.info
        COMMAND echo "backup_dir=${CMAKE_PROJECT_NAME}" >> ${install_script_dir}/version.info
        COMMAND echo "Version=${CANN_VERSION}" >> ${install_script_dir}/version.info
    )

    install(DIRECTORY ${install_script_dir}
        DESTINATION .
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
    )

    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack hccl project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack hccl project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CPACK_PACKAGE_FILE_NAME "CANN-hccl_alg-${CANN_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize/cmake/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif()
