Implementation of Kaggle Allen AI Science Challenge
Idea inspired by Cardal(winner of competition)

-to prepare input data
put training_set.tsv, validation_set.tsv and test_set.tsv into ./input

-to prepare corpus
put OEBPS/ into ./corpus/CK12
put CK-12-Biology-Concepts_b_v143_e4x_s1.text into ./corpus/CK12
put CK-12-Chemistry-Basic_b_v143_vj3_s1.text into ./corpus/CK12
put CK-12-Earth-Science-Concepts-For-High-School_b_v114_yui_s1.text into ./corpus/CK12
put CK-12-Life-Science-Concepts-For-Middle-School_b_v126_6io_s1.text into ./corpus/CK12
put CK-12-Physical-Science-Concepts-For-Middle-School_b_v119_bwr_s1.text into ./corpus/CK12
put CK-12-Physics-Concepts-Intermediate_b_v56_ugo_s1.text into ./corpus/CK12
put simplewiki-20151102-pages-articles.xml into ./corpus/simplewiki

-to run the code and generate submission:
python src/main.py
