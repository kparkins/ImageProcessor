#!/bin/bash
i=1
filename=""
while [ $i -le 5 ]; do
	filename="$i"
	filename+=$1 
	filename+=".bmp"
	./image -FloydSteinbergDither $i < Images/flower.bmp > $filename
	((i++))
done
