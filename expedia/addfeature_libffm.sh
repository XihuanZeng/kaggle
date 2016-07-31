#! /bin/sh

for ((i=0;i<=99;i++))
do
	cd ./code
	python preprocess_libffmpredict.py --cluster ${i}
	cd ..	
	./libffm/ffm-predict data/libffm_data/tmp_train_input.txt models/libffm_model0 data/libffm_data/tmp_train_output.txt
	./libffm/ffm-predict data/libffm_data/tmp_test_input.txt models/libffm_model0 data/libffm_data/tmp_test_output.txt
	cd ./code
	python preprocess_libffmaddfeature.py --cluster ${i}
	cd ..
done

