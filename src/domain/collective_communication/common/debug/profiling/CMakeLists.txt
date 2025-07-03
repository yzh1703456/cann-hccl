set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/task_profiling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_exception_handler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_runner.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/command_handle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiling_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_overflow.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_manager_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiling_manager_pub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter_prof.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dlprof_function.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dlrt_function.cc
)
 
target_sources(hccl_alg PRIVATE
    ${src_list}
)