CC = g++
CFLAGS = -O3 -Wall -Wextra -lm #-fsanitize=address -g -DDEBUG
TARGET = entropy_analyzer
SOURCES = entropy_calculator_stream.c main_stream.c
HEADERS = entropy_calculator_stream.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all debug clean
