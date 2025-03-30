CC      = gcc
CFLAGS  = -std=gnu11 -march=native -Wall -Wextra
LDFLAGS = -lm -lc
DEFINES = -DCACHE_LINE_SIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)

TARGET  = mm
DTARGET = $(TARGET)d
SRC = main.c

all: $(TARGET)

debug: $(DTARGET)

$(TARGET): $(SRC)
	$(CC) -O3 $(CFLAGS) $(DEFINES) -o $@ $^ $(LDFLAGS)

$(DTARGET): $(SRC)
	$(CC) -O0 -g $(CFLAGS) $(DEFINES) -o $@ $^ $(LDFLAGS)

run: all
	./$(TARGET)

run-priority: all
	sudo chrt -f 99 taskset -c 2,3 nice -n -20 ./$(TARGET)

clean:
	rm -f $(TARGET) $(DTARGET)

.PHONY: all run clean debug
