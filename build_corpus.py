#-*- encoding:utf-8 -*-
import re
import nltk
from time import time
from util import read_all_lines

def format_sentence(sentence):
    """
    直接去除实体标签，保留完整的句子
    @param ste ori sentence 
    @return str new sentence moved tags
    """
    sentence = re.sub('</e1>', '', sentence)
    sentence = re.sub('</e2>', '', sentence)
    sentence = re.sub('<e1>', '', sentence)
    sentence = re.sub('<e2>', '', sentence)
    return sentence

def cut_sentence(sentence, cut=True):
    """
    截取句子，以实体为界，偏移1个词
    @param ste ori sentence 
    @return str new sentence moved tags
    """
    entity_with_tag_1, entity_with_tag_2 = \
        re.findall('(<e.>.*?</e.>)', sentence)[:]
    words = sentence.split(' ')
    if ' ' in entity_with_tag_1:  # 实体1由多个单词组成
        temp = entity_with_tag_1.split(' ')[0]
        index_1 = words.index(temp)
    else:
        index_1 = words.index(entity_with_tag_1)
    if ' ' in entity_with_tag_2:  # 实体2由多个单词组成
        temp = entity_with_tag_2.split(' ')[-1]
        index_2 = words.index(temp)
    else:
        index_2 = words.index(entity_with_tag_2)
    entity_1, entity_2 = re.findall('<e.>(.*?)</e.>', entity_with_tag_1)[0], \
                         re.findall('<e.>(.*?)</e.>', entity_with_tag_2)[0]
    if cut:
        index_1_ = index_1-1 if index_1-1>=0 else 0
        index_2_ = index_2+1 if index_2+1<=len(words) else len(words)
        sentence = format_sentence(' '.join(words[index_1_:index_2_]))
        index_1 = index_1 - index_1_
        index_2 = index_2 - index_1_
    else:
        sentence = format_sentence(' '.join(words))
    return sentence, index_1, index_2

def build_train_corpus(cut=False):
    """
    构建训练语料
    """
    file = open('corpus_handle/train.txt', 'w', encoding='utf-8')
    all_lines = read_all_lines('corpus/TRAIN_FILE.TXT')
    times = int(len(all_lines)/3)
    for i in range(times):  # 每次读取三行
        line_0, line_1, line_2 = all_lines[i*3:(i+1)*3]
        # 句子
        sentence = line_0.split('\t')[1][1:-2]
        sentence = re.sub('\s+', ' ', sentence)
        sentence, index_1, index_2 = cut_sentence(sentence, cut=cut)
        # 关系类型
        relation_type = line_1.split('(')[0]
        file.write('%s|%d|%d|%s\n' % (relation_type, index_1, index_2, sentence))
    file.close()

def init_test_rel_dict():
    """
    训练语料编号到关系类型的映射
    """
    all_lines = read_all_lines('corpus/TEST_FILE_KEY.TXT')
    num2rel_dict = nltk.defaultdict(str)
    for line in all_lines:
        num, rel = line.split('\t')[:]
        num2rel_dict[num] = rel
    return num2rel_dict

def build_test_corpus(cut=False):
    """
    构建测试语料
    """
    num2rel_dict = init_test_rel_dict()
    file = open('corpus_handle/test.txt', 'w', encoding='utf-8')
    all_lines = read_all_lines('corpus/TEST_FILE.txt')
    for line in all_lines:
        items = line.split('\t')[:]
        num, sentence = items[0], items[1][1:-2]
        sentence = re.sub('\s+', ' ', sentence)
        sentence, index_1, index_2 = cut_sentence(sentence, cut=cut)  # 截取句子
        relation_type = num2rel_dict[num]
        file.write('%s|%d|%d|%s\n' % (relation_type, index_1, index_2, sentence))
    file.close()

if __name__ == '__main__':
    t0 = time()
    cut = False
    build_train_corpus(cut=cut)
    build_test_corpus(cut=cut)
    print('Done in %.2fs!' % (time()-t0))
