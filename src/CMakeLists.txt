set(SOURCES
    entropy_calculator_stream.c
    main_stream.c
)

set(HEADERS
    entropy_calculator_stream.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>: -O3 -Wall -Wextra>
    $<$<CONFIG:Debug>: -Wall -Wextra -fsanitize=address -g -DDEBUG>
)

target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>: -fsanitize=address>
)
