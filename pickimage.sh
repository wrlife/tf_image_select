#!/bin/bash

printf 'data/%s 1\n' good/*.jpg >> traintmp.txt
printf 'data/%s 0\n' bad/*.jpg >> traintmp.txt

sort -R traintmp.txt > train.txt
rm traintmp.txt
