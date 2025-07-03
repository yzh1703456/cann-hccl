set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_root_scatter_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_double_ring_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_ring_concurrent_direct.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_ring.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)
