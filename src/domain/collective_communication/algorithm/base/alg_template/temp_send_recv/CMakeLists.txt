set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/send_receive.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)
