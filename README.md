[<p align="center"><img src="qic.png" width="100"></p>](https://titaschanda.github.io/QIClib)


# [Quantum Information and Computation library (QIClib)](https://titaschanda.github.io/QIClib)
Version 1.0 - March 20, 2017 
=================================
[QIClib](https://titaschanda.github.io/QIClib) is a mordern C++11 library for general purpose quantum computing, supporting Linux, Windows and Mac OS X. 
It is a header only template library, using [Armadillo](http://arma.sourceforge.net/) (developed by Conrad Sanderson et al., Data61, Australia) for highly efficient linear algebra calculations, and if available, the [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) nonlinear optimization library for certain features.

Getting started
---------------
[QIClib](https://titaschanda.github.io/QIClib) is a header only library, so there is no need to compile the source. Download [QIClib](https://titaschanda.github.io/QIClib) either from official [website](https://titaschanda.github.io/QIClib) or using `git clone` with the command
   
     git clone https://github.com/titaschanda/QIClib.

Include the header `QIClib` in your source code (make sure that your compiler can find the path of the header file) and [QIClib](https://titaschanda.github.io/QIClib) is ready to fly. 

Make sure that you have [Armadillo](http://arma.sourceforge.net/) (version 4.2 or later) and [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) installed on your system. If you don't want to use [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt) specific features (like Quantum Discord), just add the following line before including `QIClib` header:

    #define QICLIB_DONT_USE_NLOPT

Also make sure that you have an C++11 compliant compiler. [gcc](https://gcc.gnu.org/) version 4.8 or later, or [clang](http://clang.llvm.org/) version 3.3 or later is recomended.

For example codes, see [here](https://titaschanda.github.io/QIClib/sample.html). You will also find detailed API information [here](https://titaschanda.github.io/QIClib/documentation.html).


**Note:** Instead of using standard [BLAS](http://www.netlib.org/blas/), link [OpenBLAS](http://www.openblas.net/), [Intel MKL](https://software.intel.com/en-us/intel-mkl) or [AMD ACML](http://developer.amd.com/tools-and-sdks/archive/amd-core-math-library-acml/) (or [Accelerate Framework](https://developer.apple.com/library/tvos/documentation/Accelerate/Reference/AccelerateFWRef/index.html) in MAC OSX) with latest version of [Armadillo](http://arma.sourceforge.net/) for better performance. For more see, [this](http://arma.sourceforge.net/faq.html#dependencies) and [this](https://gist.github.com/bdsatish/5646151). Also turn on compiler optimizations, e.g., in [gcc](https://gcc.gnu.org/) or [clang](http://clang.llvm.org/) add `-O3` flag during compilation. You can also add `-march=native` flag to enable [SSE3](https://en.wikipedia.org/wiki/SSE3), [SSE4](https://en.wikipedia.org/wiki/SSE4), and [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) instructions for further speed ups.


**Note:** Though older versions of [Armadillo](http://arma.sourceforge.net/) (upto version 4.2) are compatible with QIClib, it is recommended to use newer versions. If your package manager does not provide newer versions of [Armadillo](http://arma.sourceforge.net/), manually upgrade it to the latest 
[version](http://arma.sourceforge.net/download.html).

Got a Problem or Question?
--------------------------
If you have a question about how to use [QIClib](https://titaschanda.github.io/QIClib), create a new issue at [issue tracker](https://github.com/titaschanda/QIClib/issues) labelled `discussion`.

Found an Issue or Bug?
----------------------
If you found a bug in the source code or a mistake in any kind of documentation, please let us know by adding an issue to the  [issue tracker](https://github.com/titaschanda/QIClib/issues).


You are welcomed to submit a pull request with your fix afterwards, if at hand.

Requesting a Feature?
---------------------
If you are missing some features within [QIClib](https://titaschanda.github.io/QIClib), feel free to ask us about it by adding a new request to the [issue tracker](https://github.com/titaschanda/QIClib/issues) labelled `feature request`.

Note that submitting a pull request, providing the needed changes to introduced your requested feature, usually speeds up the process.

License
-------
Copyright (c) 2015 - 2019  Titas Chanda, titas DOT chanda AT gmail DOT com

[QIClib](https://titaschanda.github.io/QIClib) is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

[QIClib](https://titaschanda.github.io/QIClib) is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with [QIClib](https://titaschanda.github.io/QIClib).  If not, see <http://www.gnu.org/licenses/>.
