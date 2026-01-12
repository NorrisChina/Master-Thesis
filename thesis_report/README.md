# Content of this folder

- abstract.tex
- acronyms.tex
- Aufgabenstellung.pdf
- header.tex -> do not touch if you don't know what are you doing
- README.txt
- symbols.tex
- thesis.pdf -> PDF to get an idea how it looks like
- thesis.tex -> main file of the LaTeX template
- titlepage.tex
- appendix/appendix.tex
- bib/references.bib
- chapters/introduction.tex
- chapters/main.tex
- chapters/summary.tex
- figures/dreieck.png
- figures/Si-function.png
- figures/tiefpass.png
- figures/tikz/PLACE_HERE_YOUR_TIKZ_FILES.txt
- tables/HERE_YOU_CAN_PUT_BIG_TABLES.txt

# Requirements

- PdfLaTeX
- texlive-science
- texlive-bibtex-extra
- biber
- tested with TeXstudio 2.12.6, MiKTeX 2.9, Windows 10
- other version not tested
- For overleaf
	- header.tex -> for line 55  use (\usepackage[default,scale=0.95]{opensans})
	- Go to Menu -> set "thesis.tex" as Main document

Recommendation: Install [tex live](https://www.tug.org/texlive/quickinstall.html)

# Notes

Notice that the thesis.pdf is in the .gitignore. Please, **do not push the compiled pdf**.
After every push, the pdf is **compiled on the CI**. You can see the result here:

[Latest commented PDF online](/../-/jobs/artifacts/main/file/commented.pdf?job=build) or [Latest PDF online](/../-/jobs/artifacts/main/file/final.pdf?job=build)

---

## Container build (recommended inside this repo)

The devcontainer/Dockerfile installs the LaTeX toolchain (latexmk, biber, TUD class, and fonts). After the container builds:

- Build once:

```bash
make -C thesis_report
```

- Clean build:

```bash
make -C thesis_report clean && make -C thesis_report
```

- Watch/rebuild on changes:

```bash
make -C thesis_report watch
```

- VS Code tasks: "Thesis: Build PDF" under Terminal → Run Task…

Output: `thesis_report/thesis.pdf`.
