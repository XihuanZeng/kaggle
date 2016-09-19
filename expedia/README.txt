step 0: 
git clone https://github.com/guestwalk/libffm.git
cd libffm
make
cd ..

step 1: create ho.csv, tr.csv and te.csv which takes 100,000 users as sample. The hold out set has all event prior to 05/2014
tr.csv holds data from 05 to 10/2014 and te.csv holds data from 11 to 12/2014
cd code/
python preprocess_sampling.py

step 2: generate count features
python preprocess_featurecounter.py

step 3: expand tr.csv and te.csv for each cluster
python preprocess_traingen.py

step 4: train libffm model for each cluster on hold out set
python preprocess_libffmgen.py
bash ../train_libffm.sh

step 5: use libffm model to give score on each training and test set and append the score as feature
bash ../addfeature_libffm.sh

step 6: create input that feed into xgboost:pairwise


// New Procedure
step 1: sampling data from train.csv, select 100,000 users for analysis
python samples.py






