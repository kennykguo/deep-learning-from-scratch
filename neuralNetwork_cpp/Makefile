CC = g++
CFLAGS = -std=c++11 -Wall

# Define the source files and object files
SRCS = main.cpp Network.cpp LinearLayer.cpp Neuron.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = my_program

# Rule to compile .cpp files into .o files
%.o: %.cpp
    $(CC) $(CFLAGS) -c $< -o $@

# Rule to link the object files and create the executable
$(TARGET): $(OBJS)
    $(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Rule to clean the object files and the executable
clean:
    rm -f $(OBJS) $(TARGET)
