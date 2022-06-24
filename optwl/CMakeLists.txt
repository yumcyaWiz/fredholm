add_library(optwl INTERFACE)
target_include_directories(optwl INTERFACE
    ${OptiX_INCLUDE}
    "include/"
)
target_link_libraries(optwl INTERFACE
    spdlog::spdlog
)