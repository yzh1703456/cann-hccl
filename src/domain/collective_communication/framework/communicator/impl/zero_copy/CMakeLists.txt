set(src_list
    zero_copy_address_mgr.cc
    zero_copy_memory_agent.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)