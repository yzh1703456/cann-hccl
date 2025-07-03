set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_all_gather_double_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh_atomic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ring_concurrent_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ring_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_mesh_mix.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)