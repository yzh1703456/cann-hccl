set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_ring.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)
