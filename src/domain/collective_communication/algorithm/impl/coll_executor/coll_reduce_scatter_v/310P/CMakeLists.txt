set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_v_for_310p_ring_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)