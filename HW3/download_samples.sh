#!/bin/bash

if [ ! -d samples ]; then
	wget -O- https://www.cse.iitb.ac.in/~rdabral/CS763/CS763DeepLearningHW.tar.gz | tar xvz -C samples
fi
