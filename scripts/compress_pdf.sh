#!/bin/bash
for f in "$@"
do
  echo "${f/%.pdf/_comp.pdf}"
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile="${f/%.pdf/_comp.pdf}"  "$f"
done
