#!/bin/bash

if [ ! -d samples ]; then
	wget -O- https://www.cse.iitb.ac.in/~rdabral/CS763/CS763DeepLearningHW.tar.gz | tar -xvz
	mv "CS 763 Deep Learning HW" samples
fi
