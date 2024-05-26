# Define the default target. When you make with no arguments, this will be the target that is built.
.PHONY: default
default: build-pytriton

# Docker image version
VERSION := 0.0.0

# Docker image name
PYTRITON_IMAGE_NAME := whisper-pytriton
TENSORRT_LLM_IMAGE_NAME := whisper-tensorrt-llm
# Build the Docker image
.PHONY: build-pytriton
build-pytriton:
	@echo "Building the PyTriton Docker image..."
	docker build -f Dockerfile.pytriton -t $(PYTRITON_IMAGE_NAME):$(VERSION) .


.PHONY: build-tensorrt-llm
build-tensorrt-llm:
	@echo "Building the PyTriton Docker image..."
	docker build -f Dockerfile.tensorrt_llm -t $(TENSORRT_LLM_IMAGE_NAME):$(VERSION) .

# Help target to display help to the user
.PHONY: help
help:
	@echo "Makefile for building the PyTriton Docker image"
	@echo ""
	@echo "Usage:"
	@echo "  make build-pytriton     Build the Docker image for PyTriton"
	@echo "  make help               Display this help message"
