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
CXX = nvcc
LD = nvcc

INCLUDES = -I./hdr
CFLAGS = -g -std=c++11 -DTEST $(CUDA_CFLAGS)
LIBS = -lpthread $(CUDA_LIBS)

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
DEPS = $(OBJS:%.o=%.d)

all : $(TARGET)
	@rm -f $(OBJ_DIR)/*.o

$(TARGET) : $(OBJS)
	$(LD) -o $@ $^ $(LIBS) 

$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(CXX) $(CFLAGS) $(INCLUDES) -o $@ -c $<


.PHONY : clean
clean :
	@rm -rf $(TGT_DIR)/* $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d 

-include $(DEPS)
