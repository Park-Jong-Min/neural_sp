# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python3.7
PYTHON :=
# The python version installed in the conda setup
PYTHON_VERSION := 3.7
CUDA_VERSION := 10.1
PYTORCH_VERSION := 1.5.1
# Use a prebuild Kaldi to omit the installation
KALDI :=

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)

CONDA_PYTORCH := pytorch=$(PYTORCH_VERSION) cudatoolkit=$(CUDA_VERSION)
CUDA_DEPS := cupy.done

# Path to save tools (default: current directory)
TOOL :=
ifeq ($(strip $(TOOL)),)
	TOOL := $(PWD)
endif
CONDA=/home/jmpark/.conda/envs/asr

.PHONY: all clean

all: kaldi.done python check_install extra
python: neural_sp.done warp-ctc.done
extra: sentencepiece.done nkf.done moses.done mwerSegmenter.done

# kaldi
ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(abspath $(KALDI)) $(TOOL)/kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git $(TOOL)/kaldi
	cd $(TOOL)/kaldi/tools; $(MAKE) all
	cd $(TOOL)/kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif

# python (including upgrade check)
neural_sp.done:
	. $(CONDA)/bin/activate; pip list
	. $(CONDA)/bin/activate; pip install pip --upgrade
	. $(CONDA)/bin/activate; pip install torch==$(PYTORCH_VERSION)
	. $(CONDA)/bin/activate; pip install warp_rnnt==0.3
	. $(CONDA)/bin/activate; pip install -e ..  # setup.py
	. $(CONDA)/bin/activate && conda install -y $(CONDA_PYTORCH) -c pytorch
	touch neural_sp.done

# warp-ctc
warp-ctc.done:
	if . $(CONDA)/bin/activate && python -c 'import torch as t;major=t.__version__.split(".")[0];assert major == "1"' &> /dev/null; then \
		if [ ! -z "$(strip $(CUPY_VERSION))" ]; then \
			. $(CONDA)/bin/activate && pip install warpctc-pytorch10-cuda$(strip $(subst .,,$(CUDA_VERSION))); \
		else \
			. $(CONDA)/bin/activate && pip install warpctc-pytorch10-cpu; \
		fi \
	else \
		rm -rf warp-ctc; \
		git clone https://github.com/espnet/warp-ctc.git; $(TOOL)/warp-ctc \
		cd $(TOOL)/warp-ctc; git checkout -b pytorch-0.4 remotes/origin/pytorch-0.4; \
		mkdir build && cd build && cmake .. && $(MAKE) && cd ..; \
		. $(CONDA)/bin/activate; pip install cffi; cd pytorch_binding && python setup.py install; \
	fi
	touch warp-ctc.done

warp-transducer.done:
	rm -rf warp-transducer
	git clone https://github.com/HawkAaron/warp-transducer.git $(TOOL)/warp-transducer
	# Note: Requires gcc>=5.0 to build extensions with pytorch>=1.0
	if . $(CONDA)/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
        . $(CONDA)/bin/activate && python -c "from distutils.version import LooseVersion as V;assert V('$(GCC_VERSION)') >= V('5.0'), 'Requires gcc>=5.0'"; \
	fi
	. $(CONDA)/bin/activate; cd $(TOOL)/warp-transducer && mkdir build && cd build && cmake .. && $(MAKE); true
	. $(CONDA)/bin/activate; cd $(TOOL)/warp-transducer/pytorch_binding && python setup.py install
	touch warp-transducer.done

check_install:
	. $(CONDA)/bin/activate; python check_install.py

sentencepiece.done:
	if [ -e $(TOOL)/sentencepiece ]; then rm -rf $(TOOL)/sentencepiece; fi
	git clone https://github.com/google/sentencepiece.git $(TOOL)/sentencepiece
	cd $(TOOL)/sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	# command -v bazel > /dev/null || echo "SentencePiece requires Bazel, see https://bazel.build/"
	# cd $(TOOL)/sentencepiece && bazel build src:all --incompatible_disable_deprecated_attr_params=false
	touch sentencepiece.done

nkf.done:
	if [ -e $(TOOL)/nkf ]; then rm -rf $(TOOL)/nkf; fi
	cd $(TOOL)/nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz -P $(TOOL)/nkf
	cd $(TOOL)/nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

moses.done:
	if [ -e $(TOOL)/moses ]; then rm -rf $(TOOL)/moses; fi
	cd $(TOOL)/moses; git clone https://github.com/moses-smt/mosesdecoder.git $(TOOL)/moses
	touch moses.done

mwerSegmenter.done:
	cd $(TOOL); wget https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz -P $(TOOL)/
	cd $(TOOL); tar zxvf mwerSegmenter.tar.gz
	cd $(TOOL); rm mwerSegmenter.tar.gz
	touch mwerSegmenter.done
