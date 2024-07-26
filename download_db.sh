#!/bin/bash


center() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..499})"
  printf '\n%*.*s %s %*.*s\n\n' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

center "DOWNLOADING THE DATASET"
base_url="https://ninapro.hevs.ch/files/DB1/Preprocessed"

mkdir data
cd data

for i in {1..27}; do
    file_name="s${i}.zip" 
    file_url="${base_url}/${file_name}"
    
    wget "${file_url}"
    unzip "${file_name}" -d "s${i}"	
done

