set(alg_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_crc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_consistentcy_checker.cc
)

target_sources(hccl_alg PRIVATE
    ${alg_src_list}
)