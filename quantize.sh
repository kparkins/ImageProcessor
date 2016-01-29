#!/bin/bash
i=1
filename=""
while [ $i -le 1 ]; do
	filename="$i"
	filename+=$1 
	filename+=".bmp"
	./image -edgeDetect $i < Images/checkerboard.bmp > $filename
	i=$((i+2))
done
