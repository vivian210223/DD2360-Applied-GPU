HIPCC=hipcc

SOURCES = HelloWorld.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./HelloWorld

all: $(EXECUTABLE)

CXXFLAGS =--offload-arch=gfx90a
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
