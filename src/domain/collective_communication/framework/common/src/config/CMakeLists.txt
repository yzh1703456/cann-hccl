set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/env_config.cc
)


target_sources(hccl PRIVATE
    ${src_list}
)