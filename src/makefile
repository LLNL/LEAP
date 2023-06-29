ifeq ($(OS),Windows_NT)
        LIB_PATH = -L/usr/local/lib
        LIBS = -lfftw3 -lm

        DYLIBFLAGS = -shared
        DYLIBFILETYPE = dll
        IMPORTLIBRARYFLAGS = -Wl,--out-implib,libshared_dll.lib

	CC = c++
else
        #LIB_PATH = -L/usr/local/lib
        #LIBS = -lfftw3 -lm

	INC_PATH = -I/usr/local/include -I/usr/tce/packages/cuda/cuda-10.2.89/include
        LIB_PATH = -L/usr/local/lib -L/usr/tce/packages/cuda/cuda-10.2.89/nvidia/targets/x86_64-linux/lib

        #LIB_PATH = -L/usr/local/lib -L/usr/local/cuda-11.4/lib64
        LIBS = -lfftw3 -lm -lcudart -lcuda

        CUDA_LIB_PATH = -L/usr/local/cuda-11.4/lib64
        CUDA_LIBS = -lcudart -lcuda

        DYLIBFLAGS = -fPIC -shared
        DYLIBFILETYPE = so

	CC = g++
endif

CFLAGS = -Wall -pedantic -Wno-long-long -Wno-deprecated -fopenmp
DFlags = -g
OFLAGS = -O3

CUDA_FLAGS = -std=c++11 #-c -arch=sm_20
CUDA_CC = nvcc

BIN_DIR = bin
LIB_DIR = lib
SRC_DIR = .

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)

OBJECTS := $(patsubst $(SRC_DIR)/%,%,$(SOURCES))
OBJECTS := $(patsubst %.cpp,%.o,$(OBJECTS))
OBJECTS := $(addprefix $(BIN_DIR)/,$(OBJECTS))

CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%,%,$(CUDA_SOURCES))
CUDA_OBJECTS := $(patsubst %.cu,%.o,$(CUDA_OBJECTS))
CUDA_OBJECTS := $(addprefix $(BIN_DIR)/,$(CUDA_OBJECTS))

EXECUTABLE=$(LIB_DIR)/libprojectors.$(DYLIBFILETYPE)
MAIN_EXECUTABLE=$(BIN_DIR)/projectors

#all:build $(EXECUTABLE) $(MAIN_EXECUTABLE)
all:build $(EXECUTABLE)

$(EXECUTABLE):  $(OBJECTS) $(CUDA_OBJECTS)
	$(CC) $(DYLIBFLAGS) $(CFLAGS) $(OFLAGS) $(DFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(CUDA_OBJECTS) $(LIB_PATH) $(LIBS) $(IMPORTLIBRARYFLAGS)
	#$(CUDA_CC) projectors.cu

#$(MAIN_EXECUTABLE):  $(OBJECTS) $(CUDA_OBJECTS)
#	$(CC) $(CFLAGS) $(OFLAGS) $(DFLAGS) -o $(MAIN_EXECUTABLE) $(OBJECTS) $(CUDA_OBJECTS) $(LIB_PATH) $(LIBS)
#	#$(CUDA_CC) projectors.cu

build:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(LIB_DIR)

#$(BIN_DIR)/%.o : $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
#       @echo $<
#	$(CC) $(DYLIBFLAGS) $(CFLAGS) $(OFLAGS) $(DFLAGS) -c $< $(LIB_PATH) $(LIBS) -o $@

$(BIN_DIR)/%.o : $(SRC_DIR)/%.cpp
	@echo $<
	$(CC) $(DYLIBFLAGS) $(CFLAGS) $(OFLAGS) $(DFLAGS) -c $< $(LIB_PATH) $(LIBS) -o $@

$(BIN_DIR)/%.o : $(SRC_DIR)/%.cu
	@echo $<
	$(CUDA_CC) $(CUDA_FLAGS) --compiler-options '-fPIC' -c $< $(CUDA_LIB_PATH) $(CUDA_LIBS) -o $@

clean:
	rm -rf $(EXECUTABLE) $(MAIN_EXECUTABLE) $(OBJECTS) $(CUDA_OBJECTS) #$(BIN_DIR)
	find . -name "*~" -exec rm {} \;
	find . -name "*.o" -exec rm {} \;
	