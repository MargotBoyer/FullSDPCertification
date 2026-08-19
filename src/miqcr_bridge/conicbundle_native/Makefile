# Makefile — construit libcb_native.so (pont ctypes vers la vraie librairie
# ConicBundle, Helmberg/Kiwiel — cf. cb_wrapper.py et task-dynamic-conic-bundle.md
# "Pivot : abandon du bundle from-scratch, pont direct vers ConicBundle").
#
# Usage (depuis ce dossier, ou `make -C src/miqcr_bridge/conicbundle_native`
# depuis la racine du repo) :
#   make
#
# Ne recompile rien de ConicBundle lui-même : lie simplement en .so les objets
# déjà présents dans libcb.a (les symboles C de cb_cinterface.h y sont déjà,
# dans CBSolver.o — vérifié via `nm`), aucun wrapper C nécessaire.

CBDIR   = ../../../Miqcr-1.0_triang_sbb_gurobiter_QCP5/ConicBundle
LIBCB   = $(CBDIR)/lib/libcb.a
TARGET  = libcb_native.so

LD = g++

all: $(TARGET)

$(TARGET): $(LIBCB)
	$(LD) -shared -fPIC -o $@ \
	    -Wl,--whole-archive $(LIBCB) -Wl,--no-whole-archive \
	    -lstdc++ -lm
	@echo ">>> $(TARGET) compilée avec succès."

clean:
	rm -f $(TARGET)

.PHONY: all clean
