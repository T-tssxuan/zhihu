# Data Info

## length

```
question_train_character_desc_set: 
    max(25698), min(1), mean(117), std(226)
    [1.0, 1.0, 1.0, 8.0, 27.0, 47.0, 74.0, 106.0, 165.0, 291.0]

question_train_character_title_set: 
    max(292), min(1), mean(22), std(22)
    [1.0, 11.0, 13.0, 15.0, 17.0, 20.0, 22.0, 25.0, 30.0, 39.0]

question_train_word_desc_set: 
    max(2787), min(1), mean(58), std(118)
    [1.0, 1.0, 1.0, 2.0, 12.0, 22.0, 35.0, 51.0, 79.0, 148.0]
    
question_train_word_title_set: 
    max(187), min(1), mean(12), std(6)
    [1.0, 6.0, 8.0, 9.0, 10.0, 11.0, 13.0, 15.0, 18.0, 23.0]

each topic element numbers: 
    max(66259), min(1636), sum(7022750), mean(3517), std(4139)
    [1636, 1846, 1917, 2006, 2130, 2301, 2574, 2953, 3733, 5512]

topic_cout:
    max(19), min(1), mean(2.3), std(1)
    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
    all_leave: 778168
    all_not_leave: 851451
    the_two: 1370348
```

## word statistics
- total: 
- max_df: 0.5, stop_words: 2
- max_df: 0.5, max_features: 50000, stop_words: 470670
- max_df: 0.5, min_df: 50, stop_words: 452551
- max_df: 0.5, min_df: 10, stop_words: 368022
-

## Fasttext Word
1. 100 epcho, 128 dim, top 5, test_score: 0.30873630009518516, eval_score: 0.340780112005972, model_loss: 3.733466
2. 100 epcho, 128 dim, top 4, test_score: 0.29067849355971903, eval_score: 0.313753920186513, model_loss: 3.733466
3. 150 epcho, 128 dim, top 5, test_score: 0.2970169906346943, eval_score: 0.329263720847297, model_loss: 3.526699
4. 5 epoch, 100 dim, top 5, test_score: 0.329319208558269, eval_score: 0.362809997222741, model_loss: 5.591572, lr: 0.5
5. 10 epoch, 128 dim, top 5, minLable: 2, test_score: 0.33229068444181215, eval_score: 0.362180453118447, model_loss: 5.361451, lr: 0.2,
6. -epoch 10 -thread 30 -dim 128 -lr 0.2 -lrUpdateRate 200 -ws 10 -neg 5 -minCount 5 -minCountLabel 3 -loss softmax, eval_score: 0.33068042537208564, model_loss: 5.038941
7. -epoch 12 -thread 30 -dim 128 -lr 0.2 -lrUpdateRate 200 -ws 10 -neg 5 -minCount 5 -minCountLabel 3 -loss softmax, eval_score: 0.33215599422047204, model_loss: 4.926169
8. -epoch 12 -thread 30 -dim 128 -lr 0.2 -lrUpdateRate 200 -ws 10 -neg 6 -minCount 5 -minCountLabel 5 -loss softmax, eval_score: 0.3313259644993264, model_loss: 4.906177

## Fasttext Char
1. 200 epcho, 256 dim, top 5
