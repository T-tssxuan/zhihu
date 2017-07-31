echo ''
python3 -m zhihu.models.fasttext_char.train 5 30 128 0.8 100
python3 -m zhihu.models.fasttext_char.train 5 30 256 0.8 100
python3 -m zhihu.models.fasttext_char.train 5 30 512 0.8 100

echo ''
python3 -m zhihu.models.fasttext_char.train 10 30 512 1 100
python3 -m zhihu.models.fasttext_char.train 5 30 128 0.5 100
python3 -m zhihu.models.fasttext_char.train 5 30 128 0.8 500
python3 -m zhihu.models.fasttext_char.train 5 30 128 0.8 1000

echo ''
python3 -m zhihu.models.fasttext_char.train 10 30 128 0.8 100
python3 -m zhihu.models.fasttext_char.train 20 30 128 0.8 100
python3 -m zhihu.models.fasttext_char.train 30 30 128 0.8 500

echo ''
python3 -m zhihu.models.fasttext_char.train 5 30 128 1 100
python3 -m zhihu.models.fasttext_char.train 5 30 128 1 500
python3 -m zhihu.models.fasttext_char.train 50 30 128 0.8 500
python3 -m zhihu.models.fasttext_char.train 100 30 128 0.8 200

echo ''
python3 -m zhihu.models.fasttext_char.train 10 30 256 0.5 100
python3 -m zhihu.models.fasttext_char.train 20 30 256 0.5 300
python3 -m zhihu.models.fasttext_char.train 30 30 256 0.5 400
python3 -m zhihu.models.fasttext_char.train 50 30 256 0.5 500
python3 -m zhihu.models.fasttext_char.train 40 30 512 0.5 600

echo ''
python3 -m zhihu.models.fasttext_char.train 30 30 256 0.5 50
python3 -m zhihu.models.fasttext_char.train 30 30 256 0.5 100
python3 -m zhihu.models.fasttext_char.train 30 30 256 0.5 200
python3 -m zhihu.models.fasttext_char.train 30 30 256 0.5 300
python3 -m zhihu.models.fasttext_char.train 30 30 256 0.5 500

echo ''
python3 -m zhihu.models.fasttext_char.train 150 30 100 0.8 500
python3 -m zhihu.models.fasttext_char.train 100 30 256 0.5 200
python3 -m zhihu.models.fasttext_char.train 200 30 100 1 300
python3 -m zhihu.models.fasttext_char.train 200 30 512 1 500
