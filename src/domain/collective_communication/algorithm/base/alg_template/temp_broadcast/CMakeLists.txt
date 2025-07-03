set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/bcast_halvingdoubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/bcast_recursive_halvingdoubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_star.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)