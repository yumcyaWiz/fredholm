add_executable(hdr
    "hdr.cpp"
)
target_link_libraries(hdr PRIVATE
    fredholm
    stb_image_write
)

add_executable(test
    "test.cpp"
)