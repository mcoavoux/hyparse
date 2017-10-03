#!/usr/bin/make

PROJ_HOME=../..

#Linux/macosx
CC=g++
#Macosx
#CC=c++

ifeq ($(CC),c++)
	CC=clang++
endif

BUILD_DIR = $(PROJ_HOME)/build

NN_TRAINER_NAME = nnt
NN_PARSER_NAME = nnp


SHARED_OBJ_FILES = ../shared/forest.o ../shared/globals.o ../shared/lexer.o ../shared/sparse_encoder.o ../shared/srgrammar.o ../shared/state_graph.o ../shared/str_utils.o ../shared/treebank.o ../shared/utilities.o ../shared/sparse.o ../shared/dynamic_oracle.o
#CORE_OBJ_FILES = dense_encoder.o dense.o feedforward_neuralnet.o nnparser.o nn_utils.o lookup_table.o max_ent.o lookup_layer.o
#CORE_OBJ_FILES = dense.o dense_encoder.o lookup_layer.o neuralnet.o nnparser.o neural_net.o
#CORE_OBJ_FILES = dense.o dense_encoder.o nnparser.o neural_net.o layers.o
CORE_OBJ_FILES = dense_encoder.o nnparser.o neural_net.o layers.o character_convolution.o


NN_TRAINER_OBJ_FILES = $(SHARED_OBJ_FILES) $(CORE_OBJ_FILES) nnt.o
NN_PARSER_OBJ_FILES =  $(SHARED_OBJ_FILES) $(CORE_OBJ_FILES) nnp.o

LOG_FLAGS = -DENABLE_LEARN_LOGGER -DENABLE_PARSE_LOGGER -DNDEBUG

INSTALL_DIR = /usr/local/bin

DOXYGEN = doxygen
DOC_DIR = $(PROJ_HOME)/doc

#ifeq ($(CC),clang++)
	#INCLUDES = -I$(PROJ_HOME)/lib
	#LIBS =  -stdlib=libc++
	#CFLAGS = -O3 -Wall -std=c++11 -stdlib=libc++ $(LOG_FLAGS)
	#ALL = $(GSS_TRAINER_NAME) $(GSS_PARSER_NAME)       
#else
INCLUDES = -I$(PROJ_HOME)/lib/gtest/include -I$(PROJ_HOME)/lib -I$(PROJ_HOME)/src/shared -I$(PROJ_HOME)/src/sparse_parser -I$(PROJ_HOME)/src/dense_parser
CFLAGS = -Wall -Wno-sign-compare -std=c++11 -O3 -DGTEST_USE_OWN_TR1_TUPLE=1 $(LOG_FLAGS) -fopenmp -DEIGEN_DONT_PARALLELIZE -DENABLE_PARALLEL ##### -DCONVOLUTION -> bug Korean
LIBS =  -L$(PROJ_HOME)/lib/gtest/lib -L$(PROJ_HOME)/src/shared -L$(PROJ_HOME)/src/sparse_parser -fopenmp
ALL =  $(NN_TRAINER_NAME) $(NN_PARSER_NAME)
#endif

all :  $(ALL)

clean:
	rm -f *.o $(BUILD_DIR)/$(NN_TRAINER_NAME) $(BUILD_DIR)/$(NN_PARSER_NAME)
	rm -f ../shared/*.o ../sparse_parser/*.o *.o

doc: 
	$(DOXYGEN) doc.cfg
	rm -Rf $(DOC_DIR)/html
	rm -Rf $(DOC_DIR)/latex
	mv html $(DOC_DIR)
	mv latex $(DOC_DIR)

$(NN_TRAINER_NAME) : $(NN_TRAINER_OBJ_FILES)
	$(CC) $(LIBS) -o $(BUILD_DIR)/$(NN_TRAINER_NAME) $(NN_TRAINER_OBJ_FILES)

$(NN_PARSER_NAME) : $(NN_PARSER_OBJ_FILES)
	$(CC) $(LIBS) -o $(BUILD_DIR)/$(NN_PARSER_NAME) $(NN_PARSER_OBJ_FILES)


%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

../shared/%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

#dense.o : dense.hpp

install: $(GSS_TRAINER_NAME) $(GSS_PARSER_NAME) $(GSS_TESTS_NAME)
	cd $(BUILD_DIR)
	cp -f $(GSS_TRAINER_NAME) $(INSTALL_DIR)/$(GSS_TRAINER_NAME)
	cp -f $(GSS_PARSER_NAME) $(INSTALL_DIR)/$(GSS_PARSER_NAME)
