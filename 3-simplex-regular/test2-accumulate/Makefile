# Makefile,
# Note: to disable the compiler variables DEBUG or VERIFY, set them to =DUMMY. To enable, set them as their name
CC := g++ -std=c++17
NVCC := nvcc
# path macros
BIN_PATH := bin
OBJ_PATH := obj
DBG_PATH := debug
SRC_PATH := src

# Compiler Macros used in code
BSIZE3DX=8
BSIZE3DY=8
BSIZE3DZ=8

TARGET_NAME := prog

TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DBG_PATH)/$(TARGET_NAME)

DP := NO
MEASURE_POWER := NO

COMPILER_DEFINES := -lnvidia-ml
ARCH=-arch sm_75
CCOBJFLAGS=-DBSIZE3DX=${BSIZE3DX} -DBSIZE3DY=${BSIZE3DY} -DBSIZE3DZ=${BSIZE3DZ} -O3 -D${MEASURE_POWER}
DBGDEFINES := -DDEBUG -DVERIFY
ifeq (${DP}, NO)
	NVCCFLAGS=${ARCH} ${COMPILER_DEFINES}
else
	NVCCFLAGS=${ARCH} ${COMPILER_DEFINES} -rdc=true -lcudadevrt -DDP
endif

CCFLAGS=${COMPILER_DEFINES}

CPP_SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cpp)))
CUDA_SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.cu)))

OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(CPP_SRC)))))
CUDA_OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(CUDA_SRC)))))

DBG_OBJ := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(CPP_SRC)))))
DBG_CUDA_OBJ := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(CUDA_SRC)))))

default: makedir all

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CC) $(CCFLAGS) $(CCOBJFLAGS) -c -o $@ $<

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CCOBJFLAGS) -MMD -c -o $@ $<


$(DBG_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CC) $(CCFLAGS) $(CCOBJFLAGS) $(DBGDEFINES) -c -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CCOBJFLAGS) $(DBGDEFINES) -MMD -c -o $@ $<


$(TARGET): $(OBJ) $(CUDA_OBJ)
	$(NVCC) ${NVCCFLAGS} -o $@ $(OBJ) $(CUDA_OBJ)

$(TARGET_DEBUG) : $(DBG_OBJ) $(DBG_CUDA_OBJ)
	$(NVCC) ${NVCCFLAGS} -o $@ $(DBG_OBJ) $(DBG_CUDA_OBJ)


# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	-@rm $(DBG_PATH)/*
	-@rm $(OBJ_PATH)/*
	-@rm $(TARGET)
	-@rm $(TARGET_DEBUG)

-include $(OBJ_PATH)/*.d
-include $(DBG_PATH)/*.d

