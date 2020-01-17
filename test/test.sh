#! /bin/bash

NUMENV=1000
NUMEPISODES=50

TARGET=$(pwd)/../agymc/src
FILE=$1.py
shift
if [[ -e $FILE ]]; then
    cp $FILE $TARGET
    cd $TARGET
    printf 'time python $FILE --num-envs %s --episodes %s %s\n' $NUMENV $NUMEPISODES $@
    time python -O $FILE --num-envs $NUMENV --episodes $NUMEPISODES $@
else
    echo "File not found.\n"
fi