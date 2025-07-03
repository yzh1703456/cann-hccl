set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/i_hccl_one_sided_service.cc
)
target_sources(hccl PRIVATE
    ${src_list}
)
