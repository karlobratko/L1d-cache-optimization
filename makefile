CC      = gcc
CFLAGS  = -std=gnu11 -march=native -O3 -Wall -Wextra
LDFLAGS = -lm -lc
DEFINES = -DCACHE_LINE_SIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)

TARGET = mm
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ $^ $(LDFLAGS)

run: all
	./$(TARGET)

run-priority: all
	sudo chrt -f 99 taskset -c 2,3 nice -n -20 ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
