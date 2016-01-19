CC=g++
CFLAGS=-DUSE_UNIX -I./Includes -g -Wall
#.SUFFIXES: .cpp

SRCS=Source/bmp.cpp Source/pixel.cpp Source/image.cpp Source/vector.cpp Source/main.cpp
OBJS=$(SRCS:.cpp=.o)

image: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lm

Source/%.o: Source/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f image
	-rm -f core*
	-rm -f Source/*.o
	-rm -f *~
	-rm -f \#*
