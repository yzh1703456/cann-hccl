set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_star.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)
