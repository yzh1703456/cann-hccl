set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/threads_guard.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)