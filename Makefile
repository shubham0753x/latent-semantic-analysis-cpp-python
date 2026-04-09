CXX = g++

# Use venv python (IMPORTANT)
PYTHON = python

CXXFLAGS = -O3 -Wall -shared -std=c++14 -fPIC -march=native -ffast-math

# Includes from pybind11 + python
INCLUDES = $(shell $(PYTHON) -m pybind11 --includes) -Icomputation

# Extension suffix (.cpython-314-x86_64-linux-gnu.so)
PYTHON_EXT = $(shell $(PYTHON)-config --extension-suffix)

# (Optional but good practice)
LDFLAGS = $(shell $(PYTHON)-config --ldflags)

# Targets
TARGETS = linear_algebra$(PYTHON_EXT) rsvd$(PYTHON_EXT) tfidf$(PYTHON_EXT)

all: $(TARGETS)

# =========================
# Build rules
# =========================

linear_algebra$(PYTHON_EXT): linear_algebra_bind.cpp computation/csr.hpp computation/dense.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

rsvd$(PYTHON_EXT): rsvd_binding.cpp computation/randomized_svd.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

tfidf$(PYTHON_EXT): tfidf_binding.cpp computation/tfidf.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# =========================
# Clean
# =========================

clean:
	rm -f *$(PYTHON_EXT)
	rm -rf __pycache__