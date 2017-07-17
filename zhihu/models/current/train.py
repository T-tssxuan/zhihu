# Basic lstm

'''
1. feed the word embedding into the lstm network get: lstmw
2. feed the character embedding into the lstm network get: lstmc
3. concat the two vector get: [lstmw, lstmw] => vec
4. feed the vec to fc neetwork and a softmax layer
5. using the loss = cross_entropy to get the result
6. every time using the top 5 as the predict result
'''
