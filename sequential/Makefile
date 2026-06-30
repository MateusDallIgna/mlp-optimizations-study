CXX       = g++
CXXFLAGS  = -O3 -march=native -ffast-math -flto
INCLUDES  = -I include -I ../data/include
LDFLAGS   =
SRCDIR    = src
OBJDIR    = obj
TARGET    = mlpSequential

SRCS      = $(wildcard $(SRCDIR)/*.cpp)
OBJS      = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

all: $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all run clean
