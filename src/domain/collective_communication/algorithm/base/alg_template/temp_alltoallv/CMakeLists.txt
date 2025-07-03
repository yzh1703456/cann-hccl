set(src_list_pub
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_pairwise.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_calculator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_staged_pairwise.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list_pub}
)