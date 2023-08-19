# cui
add_executable(cui "cui/main.cpp")
target_link_libraries(cui PRIVATE
    fredholm
)

# gui
add_executable(gui "gui/main.cpp")
target_link_libraries(gui PRIVATE
    fredholm
)