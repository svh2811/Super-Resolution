if [ -e BSDS300 ]
then 
	echo "BSD300 folder Present...exiting..."
	exit
fi

# Get BSD_100
wget http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz

tar -xvzf BSDS300-images.tgz

rm BSDS300-images.tgz 