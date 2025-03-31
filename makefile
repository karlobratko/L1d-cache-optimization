CC      = gcc
CFLAGS  = -std=gnu11 -march=native -Wall -Wextra
LDFLAGS = -lm -lc
DEFINES = -DCACHE_LINE_SIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)

TARGET   = mm
DTARGET  = $(TARGET)d
OUT_BASE = mm-data

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

run-csv: all
	./$(TARGET) -o $(OUT_BASE).csv

run-csv-priority: all
	sudo chrt -f 99 taskset -c 2,3 nice -n -20 ./$(TARGET) -o $(OUT_BASE).csv

visualize:
	@if [ ! -f $(OUT_BASE).csv ]; then \
		echo "Error: $(OUT_BASE).csv not found. Run 'make run-csv' first."; \
		exit 1; \
	fi
	python3 visualize.py -i $(OUT_BASE).csv -o $(OUT_BASE)

plot: run-csv visualize

plot-priority: run-csv-priority visualize

clean:
	rm -f $(TARGET) $(DTARGET) $(OUT_BASE)*

.PHONY: all run clean debug run-csv run-csv-priority visualize plot plot-priority
