# Define the default target now so that it is always the first target
BUILD_TARGETS = main

# Binaries only useful for tests
TEST_TARGETS = tests/test-tokenizer tests/test-text-encoder

default: $(BUILD_TARGETS)

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
#

# keep standard at C11 and C++11
# -Ofast tends to produce faster code, but may not be available for some compilers.
ifdef BARK_FAST
OPT = -Ofast
else
OPT = -O3
endif
CFLAGS   = -I. $(OPT) -std=c11   -fPIC
CXXFLAGS = -I. $(OPT) -std=c++11 -fPIC
LDFLAGS  =

ifdef BARK_DEBUG
	CFLAGS   += -O0 -g
	CXXFLAGS += -O0 -g
	LDFLAGS  += -g
else
	CFLAGS   += -DNDEBUG
	CXXFLAGS += -DNDEBUG
endif

ifdef BARK_SERVER_VERBOSE
	CXXFLAGS += -DSERVER_VERBOSE=$(BARK_SERVER_VERBOSE)
endif

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith \
			-Wmissing-prototypes
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# detect Windows
ifneq ($(findstring _NT,$(UNAME_S)),)
	_WIN32 := 1
endif

# library name prefix
ifneq ($(_WIN32),1)
	LIB_PRE := lib
endif

# Dynamic Shared Object extension
ifneq ($(_WIN32),1)
	DSO_EXT := .so
else
	DSO_EXT := .dll
endif

# Windows Sockets 2 (Winsock) for network-capable apps
ifeq ($(_WIN32),1)
	LWINSOCK2 := -lws2_32
endif

ifdef BARK_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifdef BARK_PERF
	CFLAGS   += -DGGML_PERF
	CXXFLAGS += -DGGML_PERF
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	CFLAGS   += -march=native -mtune=native
	CXXFLAGS += -march=native -mtune=native

	# Usage AVX-only
	#CFLAGS   += -mfma -mf16c -mavx
	#CXXFLAGS += -mfma -mf16c -mavx

	# Usage SSSE3-only (Not is SSE3!)
	#CFLAGS   += -mssse3
	#CXXFLAGS += -mssse3
endif

ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS   += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif

ifndef BARK_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif # BARK_NO_ACCELERATE

ifdef BARK_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS $(shell pkg-config --cflags openblas)
	LDFLAGS += $(shell pkg-config --libs openblas)
endif # BARK_OPENBLAS

ifdef BARK_BLIS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/blis -I/usr/include/blis
	LDFLAGS += -lblis -L/usr/local/lib
endif # BARK_BLIS

ifdef BARK_CUBLAS
	CFLAGS    += -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	CXXFLAGS  += -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	LDFLAGS   += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib
	OBJS      += ggml-cuda.o
	NVCCFLAGS = --forward-unknown-to-host-compiler
ifdef BARK_CUDA_NVCC
	NVCC = $(BARK_CUDA_NVCC)
else
	NVCC = nvcc
endif #BARK_CUDA_NVCC
ifdef CUDA_DOCKER_ARCH
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else
	NVCCFLAGS += -arch=native
endif # CUDA_DOCKER_ARCH
ifdef BARK_CUDA_FORCE_DMMV
	NVCCFLAGS += -DGGML_CUDA_FORCE_DMMV
endif # BARK_CUDA_FORCE_DMMV
ifdef BARK_CUDA_DMMV_X
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(BARK_CUDA_DMMV_X)
else
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # BARK_CUDA_DMMV_X
ifdef BARK_CUDA_MMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(BARK_CUDA_MMV_Y)
else ifdef BARK_CUDA_DMMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(BARK_CUDA_DMMV_Y) # for backwards compatibility
else
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=1
endif # BARK_CUDA_MMV_Y
ifdef BARK_CUDA_DMMV_F16
	NVCCFLAGS += -DGGML_CUDA_DMMV_F16
endif # BARK_CUDA_DMMV_F16
ifdef BARK_CUDA_KQUANTS_ITER
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(BARK_CUDA_KQUANTS_ITER)
else
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif
ifdef BARK_CUDA_CCBIN
	NVCCFLAGS += -ccbin $(BARK_CUDA_CCBIN)
endif
ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -Wno-pedantic -c $< -o $@
endif # BARK_CUBLAS

ifdef BARK_CLBLAST

	CFLAGS   += -DGGML_USE_CLBLAST $(shell pkg-config --cflags clblast OpenCL)
	CXXFLAGS += -DGGML_USE_CLBLAST $(shell pkg-config --cflags clblast OpenCL)

	# Mac provides OpenCL as a framework
	ifeq ($(UNAME_S),Darwin)
		LDFLAGS += -lclblast -framework OpenCL
	else
		LDFLAGS += $(shell pkg-config --libs clblast OpenCL)
	endif
	OBJS    += ggml-opencl.o

ggml-opencl.o: ggml-opencl.cpp ggml-opencl.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif # BARK_CLBLAST

ifdef BARK_METAL
	CFLAGS   += -DGGML_USE_METAL -DGGML_METAL_NDEBUG
	CXXFLAGS += -DGGML_USE_METAL
	LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
	OBJS     += ggml-metal.o
endif # BARK_METAL

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	CFLAGS   += -mcpu=native
	CXXFLAGS += -mcpu=native
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif

ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifdef BARK_METAL
ggml-metal.o: ggml-metal.m ggml-metal.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # BARK_METAL

#
# Print build information
#

$(info I bark.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

#
# Build library
#

ggml.o: ggml.c
	$(CC)   $(CFLAGS)     -c $< -o $@

encodec.o: encodec.cpp
	$(CXX)  $(CXXFLAGS)   -c $< -o $@

bark.o: bark.cpp bark.h
	$(CXX)  $(CXXFLAGS)   -c $< -o $@

clean:
	rm -vf *.o *.so *.dll encodec bark tests/test-tokenizer

bark: bark.cpp         encodec.o ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

main: examples/main.cpp  ggml.o bark.o encodec.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

#
# Test
#

tests: $(TEST_TARGETS)

tests/test-tokenizer: tests/test-tokenizer.cpp ggml.o bark.o encodec.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-text-encoder: tests/test-text-encoder.cpp ggml.o bark.o encodec.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)
