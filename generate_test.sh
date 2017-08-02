THREAD=20
# python3 -m zhihu.models.fasttext_word.train 10 $THREAD 100 0.2 200 test
# python3 -m zhihu.models.fasttext_word.train 10 $THREAD 100 0.4 300 test
# python3 -m zhihu.models.fasttext_word.train 10 $THREAD 100 0.3 100 test
# python3 -m zhihu.models.fasttext_word.train 10 $THREAD 256 0.5 100 test
python3 -m zhihu.models.fasttext_word.train 5 $THREAD 128 0.8 1000 test
python3 -m zhihu.models.fasttext_word.train 5 $THREAD 128 0.8 500 test
python3 -m zhihu.models.fasttext_word.train 5 $THREAD 128 0.5 100 test
