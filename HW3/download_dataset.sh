#!/bin/bash

if [ ! -d dataset ]; then
	echo "Creating directory dataset"
	mkdir dataset
fi

if [ ! -d dataset/Train ]; then
	echo "Creating directory dataset/Train"
	mkdir -p dataset/Train
fi

if [ ! -d dataset/Test ]; then
	echo "Creating directory dataset/Test"
	mkdir -p dataset/Test
fi

if [ ! -f dataset/Train/data.bin ]; then
	echo "Downloading Train/data.bin"
	wget https://www.cse.iitb.ac.in/~rdabral/CS763/Train/data.bin -O dataset/Train/data.bin
fi

if [ ! -f dataset/Train/labels.bin ]; then
	echo "Downloading Train/labels.bin"
	wget https://www.cse.iitb.ac.in/~rdabral/CS763/Train/labels.bin -O dataset/Train/labels.bin
fi

if [ ! -f dataset/Test/test.bin ]; then
	echo "Downloading Test/test.bin"
	wget https://www.cse.iitb.ac.in/~rdabral/CS763/Test/test.bin -O dataset/Test/test.bin
fi
