#!/bin/bash

if [ $# -eq 0 ]
  then
    exit 1
fi
char=$(bjobs -w $1)
arr=($char)
host=${arr[13]}
regex='([0-9]+\*)?(.+)'
[[ $host =~ $regex ]]
host=${BASH_REMATCH[2]}
id=$(bhosts -l -gpu $host | grep $1)
id="${id:0:1}"
bhosts -l -gpu $host | grep -B 1 -A 3 "HOST:"
bhosts -l -gpu $host | grep "MUSED"
bhosts -l -gpu $host | grep "^${id} *EXCLUSIVE"
echo ""
