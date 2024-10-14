#!/bin/bash

for i in {1..10}
do
    py main.py > output$i.txt
done