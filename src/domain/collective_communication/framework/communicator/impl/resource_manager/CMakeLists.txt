set(src_list
    transport_manager.cc
    preempt_port_manager.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)