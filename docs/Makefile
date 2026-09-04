# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

ifeq ($(OS),Windows_NT)
clean:
	- if exist "$(BUILDDIR)" rmdir /s /q "$(BUILDDIR)"
	- if exist "auto_examples" rmdir /s /q "auto_examples"
	- if exist "generated" rmdir /s /q "generated"
else
clean:
	-$(RM) -r "$(BUILDDIR)"/* 2>/dev/null || true
	-$(RM) -r auto_examples 2>/dev/null || true
	-$(RM) -r generated 2>/dev/null || true
endif
