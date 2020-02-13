#! /usr/bin/bash env

DATA_DIR="../../data/hmdb51/"

cd ${DATA_DIR}

wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x  hmdb51_org.rar
#rm hmdb51_org.rar

mv ./hmdb51_org ./videos

cd "../../data_tools/hmdb51"
