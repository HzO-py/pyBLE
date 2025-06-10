# Makefile for compiling a C file into a shared library for ctypes in Python

CC = gcc

OUTPUT_DIR = ./output

CFLAGS = -Wall -Wextra -fPIC

SRC = utility.c

Target_LIB = utility.so

TARGET_PATH = $(OUTPUT_DIR)/$(Target_LIB)

all: $(TARGET_PATH)

${TARGET_PATH}: $(SRC)
	@mkdir -p $(OUTPUT_DIR)
	@echo "Compiling $< into shared library $@"
	$(CC) -shared -o $@ $< $(CFLAGS)


clean:
	@echo "Cleaning up generated files..."
	rm -rf $(OUTPUT_DIR)

.PHONY: all clean