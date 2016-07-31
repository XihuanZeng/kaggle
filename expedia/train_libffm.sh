#! /bin/sh
cd ./models
for ((i=0;i<=99;i++))
do
	python ../code/preprocess_libffmgen.py --cluster ${i}
	../libffm/ffm-train -t 20 ../data/libffm_data/train.txt libffm_model${i}
done

