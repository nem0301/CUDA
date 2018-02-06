OBJ_DIR = ./obj
SRC_DIR = ./src
HDR_DIR = ./hdr
TGT_DIR = ./tgt
APP_NAME = test

ARCH	= compute_30
CODE	= sm_30
CUDA_CFLAGS	= -gencode arch=$(ARCH),code=$(CODE)
CUDA_LIBS	= -gencode arch=$(ARCH),code=$(CODE)

TARGET = $(TGT_DIR)/$(APP_NAME)
NCXX = nvcc
NLD = nvcc
CXX = g++
LD = g++

INCLUDES = -I./hdr -I/usr/local/cuda/include
CFLAGS = -g -DTEST -std=c++11 -DDEBUG_LVL=2
LIBS = -lpthread -lcuda -lcublas -lcudart

NSRCS = $(wildcard $(SRC_DIR)/*.cu)
NOBJS = $(NSRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

DEPS = $(OBJS:%.o=%.d)

all : $(TARGET)
	@rm -f $(OBJ_DIR)/*.o

$(TARGET) : $(OBJS) $(NOBJS)
	$(NLD) -o $@ $^ $(LIBS) $(CUDA_LIBS);

$(NOBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NCXX) $(CFLAGS) $(CUDA_CFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) $(INCLUDES) -o $@ -c $<


.PHONY : clean
clean :
	@rm -rf $(TGT_DIR)/* $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d 

-include $(DEPS)
