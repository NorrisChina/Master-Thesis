# Makefile for building thesis_report
TEX=thesis.tex
PDF=thesis.pdf

LATEXMK=latexmk
LATEXMK_OPTS=-pdf -pdflatex='pdflatex -interaction=nonstopmode -file-line-error'

all: pdf

pdf:
	$(LATEXMK) $(LATEXMK_OPTS) $(TEX)

bib:
	biber thesis
	$(LATEXMK) $(LATEXMK_OPTS) $(TEX)

watch:
	$(LATEXMK) -pvc $(LATEXMK_OPTS) $(TEX)

clean:
	$(LATEXMK) -C

distclean: clean
	rm -f $(PDF)
