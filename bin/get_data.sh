wget --no-check-certificate 'https://drive.google.com/u/0/uc?export=download&confirm=rv78&id=1L-o6r3_GF-pSaaOTvUOPsgiTKMq_dxIl' -O data.zip
unzip data.zip -d data
cd
mv bin/data/ src/data
rm data.zip
