#===================================================
# 用于将event_mention转化为对应的0-hop concept
#===================================================
from collections import defaultdict

import pickle
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer # 词元化
from nltk.corpus import wordnet

# 公共配置
ds = 'ESC' # dataset, 值应为'ESC'或者'CTB'
prefix = '混合匹配'

document_data_path = f'./data/document/document_{ds}.pickle' # 数据集初步读取结果
bert_match_path = f'./data/Matching/BERT_matching_{ds}.pickle' # Bert嵌入匹配的结果
result_save_path = f'./data/Matching/result/{prefix}_{ds}.txt' # 匹配结果保存路径

all_concept_path = './data/Matching/all_head_concept.txt' # ConceptNet中的全部concept

# 创建公用词元化模型
lemmatizer = WordNetLemmatizer() 

# 从文件中读取ConceptNet中的concept列表
with open(all_concept_path, 'r') as f:
    ALL_CONCEPT = [line.strip('\n\r') for line in f.readlines()]

# 从文件中读取BERT嵌入匹配结果
with open(bert_match_path, 'rb') as f:
    E2C_BERT_DICT = pickle.load(f)


def get_match_dict(ds='ESC', type='混合匹配'):
    """ 从文件中读取事件描述与0-hop概念的匹配结果(用于外部文件引用) """

    match_result_path = f'./data/Matching/result/{type}_{ds}.txt' # 最终匹配结果(用于外部打卡)
    with open(match_result_path, 'r') as f:
        MATCH_DICT = { }
        for line in f.readlines():
            mention, concept = line.strip('\n').split('\t')
            MATCH_DICT[mention] = concept
    return MATCH_DICT


def get_sentence_number(tid_seq, all_token):
    """根据t_id序列获得所在句的sentence_num"""

    tid = tid_seq.split('_')[0]
    for token in all_token:
        if token[0] == tid:
            return token[1]


def nth_sentence(sen_no, all_token):
    """根据sentence_num获得text列表"""

    res = []
    for token in all_token:
        if token[1] == sen_no:
            res.append(token[-1])
    return res


def get_number_list(tid_seq, all_token):
    """根据t_id序列产生number列表"""
    
    number_list = []
    for c in tid_seq.split('_'):
        token = all_token[int(c) - 1]
        number_list.append(int(token[2]))
    return number_list


def get_sentence_offset(tid_seq, all_token):
    """根据t_id序列产生number序列"""
    
    return '_'.join(get_number_list(tid_seq, all_token))


def get_wordnet_pos(treebank_tag):
    """根据nltk词性标志获得wordnet词性标志"""

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(sentence, span):
    """对事件描述进行首词词元化处理"""

    # 获取事件描述的text列表
    event_mention = [sentence[i] for i in span] 

    # 获取句子中每个单词的词性
    word_pos = pos_tag(sentence)

    # 获取事件描述的词性 (此处以首个token代表事件描述)
    event_word_pos = get_wordnet_pos(word_pos[span[0]][1]) 

    return lemmatizer.lemmatize(event_mention[0], pos=event_word_pos) 


def remove_symbol(word):
    """去除特殊符号"""

    new_word = word.strip("'\"") # 去除首尾的'和"
    return new_word


def lemma_match(mention, sentence=None, span=None):
    """
    根据sentence情况选择对事件描述进行直接词元化或上下文词元化, 获取对应词元

    :param mention: event_mention, 事件描述
    :param sentence: 事件所在句的text列表
    :param span: 事件描述在句子中的位置
    :return: 匹配成功则返回词元(即0-hop概念), 匹配失败则返回None
    """

    tokens = mention.lower().split('_')
    # 对仅包含1个token的事件描述进行词元化处理
    if len(tokens)==1:
        lemma = None
        # # 若给定上下文, 则进行上下文词元化
        if sentence:
            lemma = lemmatize(sentence, span)
            lemma = remove_symbol(lemma)
        # 若上下文词元化匹配失败, 则进行直接词元化
        if lemma not in ALL_CONCEPT:
            lemma = lemmatize(tokens, [0])
            lemma = remove_symbol(lemma)
        # 返回匹配结果
        return lemma if lemma in ALL_CONCEPT else None
    else:
        return None


def mention_to_concept(mention, sentence=None, span=None):
    """ 根据event_mention`(以'_'为分隔符)`返回与之匹配的0-hop概念 """

    # # # 1. BERT嵌入+余弦相似性匹配
    # zero_concept = E2C_BERT_DICT[mention]

    # # # 2. 先直接匹配(498/1621), 再进行BERT嵌入+余弦相似性匹配
    # zero_concept = mention if mention in ALL_CONCEPT else E2C_BERT_DICT[mention]

    # # # 2. 直接匹配+词元化匹配
    # zero_concept = None
    # if mention in ALL_CONCEPT:
    #     zero_concept = mention
    # else:
    #     token = sentence[span[0]].lower()
    #     pos = get_wordnet_pos(token) or wordnet.NOUN
    #     zero_concept = remove_symbol(lemmatizer.lemmatize(token, pos))

    # # # 3. 先首词词元化匹配, 再进行BERT嵌入+余弦相似性匹配
    # tokens = mention.split('_')
    # lemma = remove_symbol(lemmatizer.lemmatize(tokens[0]))
    # zero_concept = lemma if lemma in ALL_CONCEPT else E2C_BERT_DICT[mention]

    # # 4. 先直接匹配, 再进行单词词元化匹配, 最后进行Bert嵌入匹配
    # # 直接词元化: {'总数': 1621, '直接匹配': 486, '词元化匹配': 71, 'BERT匹配': 1064}
    # # 改进词元化+上下文词元化: {'总数': 1626, '直接匹配': 486, '词元化匹配': 301, 'BERT匹配': 839}
    # # 改进词元化+上下文词元化(CTB数据集): {'总数': 2133, '直接匹配': 682, '词元化匹配': 395, 'BERT匹配': 1056}
    zero_concept = None
    if mention in ALL_CONCEPT:
        zero_concept = mention
    else:
        lemma = lemma_match(mention, sentence, span)
        zero_concept = lemma if lemma else E2C_BERT_DICT[mention]
    return zero_concept


if __name__ == '__main__':

    # 读取配置文件
    with open(document_data_path, 'rb') as f:
        documents = pickle.load(f)

    # ============================================================
    # 遍历所有事件描述, 根据sentence与span生成0-hop概念, 最终保存
    # ============================================================

    # 针对所有event_mention获取对应的0-hop概念
    mention_concept_dict = defaultdict(set)
    match_type_count = {'总数':0, '直接匹配':0, '词元化匹配':0, 'BERT匹配':0}
    for doc in documents:
        [all_token, event_dict, _, _] = documents[doc]
        for m_id in event_dict:
            sen_num = get_sentence_number(event_dict[m_id], all_token) # 事件所在句的sentence_num
            sentence = nth_sentence(sen_num, all_token) # 事件所在句的token_list
            span = get_number_list(event_dict[m_id], all_token) # 事件描述在句中的位置
            mention = '_'.join([sentence[i] for i in span]).lower() # 事件描述
            if ds=='CTB': mention = mention.replace('-', '_')
            concept = mention_to_concept(mention, sentence, span)
            match_type = -1
            if concept not in mention_concept_dict[mention]:
                match_type_count['总数'] += 1
                if mention==concept:
                    match_type_count['直接匹配'] += 1
                elif lemma_match(mention, sentence, span)==concept:
                    match_type_count['词元化匹配'] += 1
                elif concept==E2C_BERT_DICT[mention]:
                    match_type_count['BERT匹配'] += 1
                    match_type = 3
            if match_type==3 and len(mention_concept_dict[mention])>0:
                match_type_count['总数'] -= 1
                match_type_count['BERT匹配'] -= 1
                continue
            mention_concept_dict[mention].add(concept)
    print(match_type_count)

    # 统计同样的event_mention具有多个0-hop概念的情况
    count_gt2 = 0
    for k, v in mention_concept_dict.items():
        if len(v)>=2: 
            count_gt2 += 1
            print(k, v)
    print(count_gt2)

    # # 根据统计结果在txt文件中对匹配数据进行修正
    # # injured {'injure', 'injury'} -> 'injure'
    # # dies {'kills', 'die'} -> 'die'
    # # injures {'injure', 'pulling_incurring_injury'} -> 'injure'
    # # rioting {'picketing', 'riot'} -> 'riot'
    # # destroys {'destroying', 'destroy'} -> 'destroy'
    if ds=='CTB':
        mention_concept_dict['trading'] = {'trade'}
        mention_concept_dict['ruling'] = {'rule'}
    
    # # 保存数据
    with open(result_save_path,'w') as f:
        for k, v in mention_concept_dict.items():
            concepts = '\t'.join(v)
            f.write(f'{k}\t{concepts}\n')