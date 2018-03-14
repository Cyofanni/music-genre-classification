#!bin/sh

INPUT_DIR=$1

for file in $(ls $INPUT_DIR)
do
   old_extension="${file##*.}"
   filename_no_ext="${file%.*}" 
   new_file="${filename_no_ext}.wav"

   ffmpeg -i "${INPUT_DIR}/$file" "${INPUT_DIR}/$new_file"
done
