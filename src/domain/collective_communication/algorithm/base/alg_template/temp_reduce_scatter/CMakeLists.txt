set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_reduce_scatter_double_ring_with_serial_local_copy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/aligned_reduce_scatter_double_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_atomic_opbase.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_atomic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_recursive_hd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ring_concurrent_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_mix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_mesh_mix_single_stream.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)
