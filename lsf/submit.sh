#!/bin/bash
sed "s|ARGS|${*:2}|g" ${1} | bsub
