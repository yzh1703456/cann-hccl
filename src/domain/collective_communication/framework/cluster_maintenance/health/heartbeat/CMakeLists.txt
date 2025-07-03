set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/heartbeat.cc
)


target_sources(hccl PRIVATE
    ${src_list}
)