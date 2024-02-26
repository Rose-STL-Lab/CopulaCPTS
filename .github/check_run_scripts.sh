#!/bin/bash

rm filechk_results.txt

declare -a FilesArray=("requirements.txt" "run_experiment.sh")

for val in ${FilesArray[@]}; 
do
    echo "Checking presence of $val"
    if test -f "$val"; 
    then
        echo "$val exists" >> filechk_results.txt
    else    
        echo "$val does not exist" >> filechk_results.txt
    fi
done
