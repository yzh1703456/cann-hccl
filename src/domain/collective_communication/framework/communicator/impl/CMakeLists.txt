set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator_attrs.cc
    task_abort_handler.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)

add_subdirectory(resource_manager)
add_subdirectory(one_sided_service)
add_subdirectory(zero_copy)