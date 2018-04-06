if [ -e BSD100_SR ]
then 
	echo "BSD100_SR folder Present..."
else
	echo "Creating folder BSD100_SR..."
	mkdir BSD100_SR
fi

cd BSD100_SR 

# Get BSD_100
wget https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip

unzip qgctsplb8txrksm9to9x01zfa4m61ngq.zip

rm qgctsplb8txrksm9to9x01zfa4m61ngq.zip
