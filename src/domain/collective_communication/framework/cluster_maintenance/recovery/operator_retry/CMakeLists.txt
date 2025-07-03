file(GLOB_RECURSE OPRETRY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
set(src_list
    ${OPRETRY_SRC}
)

target_sources(hccl PRIVATE
    ${src_list}
)
