#===================================================
#读取Causal-TimeBank数据集, 生成document_CTB.pickle
#===================================================

import collections
import os
import os.path
import pickle
import re

from lxml import etree


def all_tokens(filename):
    """获取所有token的信息, 返回列表中的元素类型为元组, 即(t_id, sentence, number, text)"""

    ecbplus = etree.parse(filename, etree.XMLParser(remove_blank_text=True))
    root_ecbplus = ecbplus.getroot()
    root_ecbplus.getchildren()
    all_token = []
    for elem in root_ecbplus.findall('token'):
        temp = (elem.get('id'), elem.get('sentence'), elem.get('number'), elem.text)
        all_token.append(temp)

    sent_first_number = {} # key为句子序号, value为句子中首个单词的id
    for i, token in enumerate(all_token):
        id, sentence, number, text = token
        if sentence not in sent_first_number:
            sent_first_number[sentence] = id
        number = str(int(id)-int(sent_first_number[sentence]))
        all_token[i] = (id, sentence, number, text)

    return all_token


def extract_event_CAT(etreeRoot):
    """提取事件, 返回事件词典 (key为事件id, value为事件的t_id列表)"""

    event_dict = collections.defaultdict(list)
    for elem in etreeRoot.findall('Markables/'):
        if elem.tag == "EVENT":
            event_mention_id = str(elem.get('id', 'nothing'))
            for token_id in elem.findall('token_anchor'):
                token_mention_id = token_id.get('id', 'nothing')
                event_dict[event_mention_id].append(token_mention_id)
    return event_dict


def extract_plotLink(etreeRoot, event_dict):
    """提取事件对之间的关系, 返回事件关系词典 (key为事件对组成的元组即(e1,e2), value为关系)"""

    plot_dict = list()
    for elem in etreeRoot.findall('Relations/'):
        if elem.tag == "CLINK":
            source_pl = elem.find('source').get('id', 'null')
            target_pl = elem.find('target').get('id', 'null')
            rel = elem.get('relType', 'null')
            if source_pl in event_dict:
                val1 = "_".join(event_dict[source_pl])
                if target_pl in event_dict:
                    val2 = "_".join(event_dict[target_pl])
                    if int(event_dict[source_pl][0]) < int(event_dict[target_pl][0]):
                        rel = rel if rel!='null' else 'PRECONDITION'
                        plot_dict.append([val1, val2, rel])
                    else:
                        rel = rel if rel!='null' else 'FALLING_ACTION'
                        plot_dict.append([val2, val1, rel])

    return plot_dict


def read_file(ecbstart_new):
    """读取ECB文件与评估文件, 返回(事件词典, 事件关系词典, 评估数据)"""

    ecbstar = etree.parse(ecbstart_new, etree.XMLParser(remove_blank_text=True))
    ecbstar_root = ecbstar.getroot()
    ecbstar_root.getchildren()
    ecb_star_events = extract_event_CAT(ecbstar_root) # 提取所有事件 (事件m_id: 事件t_id列表(之后会被转换为t_id序列)) 
    ecbstar_events_plotLink = extract_plotLink(ecbstar_root, ecb_star_events) # 提取数据文件中所有事件关系
    return ecb_star_events, ecbstar_events_plotLink


def make_corpus(data_path, datadict):
    """
    根据指定路径制作语料库
    
    :param ecbstartopic: 初始数据文件路径
    :param evaluationtopic: 评估文件路径
    :param datadict: 全局词典, key为文档名, value为[token词典, 事件词典, plotLink列表, evalData列表]
    """
    
    for xml_file in os.listdir(data_path):
        xml_path = os.path.join(data_path, xml_file)
        ecb_star_events, ecbstar_events_plotLink = read_file(xml_path)
            
        for key in ecb_star_events: # 将t_id列表转化为t_id序列
            ecb_star_events[key] = '_'.join(ecb_star_events[key])

        all_token = all_tokens(xml_path)
        datadict[xml_file] = [all_token, ecb_star_events, None, ecbstar_events_plotLink]


def extract_event_tokens(t_id_seq, all_tokens):
    """根据t_id序列获取对应的text序列"""

    tokens = []
    for t_id in t_id_seq.split('_'):
        text = all_tokens[int(t_id)-1][-1]
        text = text.replace('-','_')
        tokens.append(text)
    return '_'.join(tokens)


def save_event_mention_txt(data_dict, txt_path=None):
    """ 提取所有event_mention, 写入txt """

    # 使用dict中的key来记录text_seq (作用等于set)
    # 使用dict中的value来记录每个事件所属文件 (用于对异常事件进行定位)
    event_set_uncased = collections.defaultdict(set)

    for file in data_dict:
            all_tokens, event_dict = data_dict[file][:2]
            for m_id in event_dict:
                text_seq = extract_event_tokens(event_dict[m_id], all_tokens).lower()
                event_set_uncased[text_seq].add(file)

    with open(txt_path, 'w') as f:
        for k in event_set_uncased:
            f.write(f'{k}\n')

def main():
    data_path = './data/Causal-TimeBank'
    data_dict = {}

    make_corpus(data_path, data_dict)

    # 保存所有event_mention
    save_event_mention_txt(data_dict, './data/Matching/all_CTB_event.txt')

    # 保存读取到的数据
    with open('./data/document/document_CTB.pickle', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
