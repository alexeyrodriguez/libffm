CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

OMP_DFLAG = -DUSEOMP
OMP_CXXFLAGS = -fopenmp

DFLAG += $(OMP_DFLAG)
CXXFLAGS += $(OMP_CXXFLAGS)

all: ffm-train ffm-predict

ffm-train: ffm-train.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm-predict: ffm-predict.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f ffm-train ffm-predict ffm.o
