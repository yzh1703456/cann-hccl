set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_deter_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_fast_double_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_aligned_reduce_scatter_double_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_ring_for_910_93_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mesh_dma_elimination.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mesh_opbase_pipeline_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_single_rank_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_double_ring_concurrent_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mix_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mesh_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mesh_aiv_smallcount_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_semi_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_aiv_rdma_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_scatter_mesh_graph_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(310P)