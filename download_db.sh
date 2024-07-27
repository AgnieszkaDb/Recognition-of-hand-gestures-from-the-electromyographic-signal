#!/bin/bash


center() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\n%*.*s %s %*.*s\n\n' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}


DIR="data"

if [ -d "$DIR" ]; then
  echo -e "Directory $DIR already exists. \n"
else
  echo -e "Directory $DIR does not exist. \n"
  center "DOWNLOADING THE DATASET"
  mkdir data
  cd data

  base_url="https://ninapro.hevs.ch/files/DB1/Preprocessed"
  for i in {1..27}; do
      file_name="s${i}.zip" 
      file_url="${base_url}/${file_name}"
      
      wget "${file_url}"
      unzip "${file_name}" -d "s${i}"	
  done
fi

