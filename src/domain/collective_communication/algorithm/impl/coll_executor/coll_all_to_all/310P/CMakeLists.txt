set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_to_all_v_for_310p_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)