#!/bin/bash 

for file in /dataset/human-action/*/*.avi; 
do 
filename="$(basename "${file}" .avi)"
mkdir "/dataset/human-action/images/${filename}"
ffmpeg -i "$file" "/dataset/human-action/images/${filename}/${output-%05d}".jpg;
done;