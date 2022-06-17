#===================================================
#读取EventLineStory数据集, 生成document_raw.pickle
#===================================================

import os
import os.path
from lxml import etree
import collections
import pickle

tag_set = set() # 统计数据集中tag类型

def read_evaluation_file(fn):
    """读取评估文件 (真实标签), 返回的评估数据列表中每个元素类型为元组, 即(e1的tid序列, e2的tid序列, 关系)"""

    res = []
    if not os.path.exists(fn):
        return res
    for line in open(fn):
        fileds = line.strip().split('\t')
        res.append(fileds)
    return res


def all_tokens(filename):
    """获取所有token的信息, 返回列表中的元素类型为元组, 即(t_id, sentence, number, text)"""

    ecbplus = etree.parse(filename, etree.XMLParser(remove_blank_text=True))
    root_ecbplus = ecbplus.getroot()
    root_ecbplus.getchildren()
    all_token = []
    for elem in root_ecbplus.findall('token'):
        temp = (elem.get('t_id'), elem.get('sentence'), elem.get('number'), elem.text)
        all_token.append(temp)
    return all_token


def extract_event_CAT(etreeRoot):
    """提取事件, 返回事件词典 (key为事件id, value为事件的t_id列表)"""

    event_dict = collections.defaultdict(list)
    for elem in etreeRoot.findall('Markables/'):
        if elem.tag.startswith("ACTION") or elem.tag.startswith("NEG_ACTION"):
            event_mention_id = elem.get('m_id', 'nothing')
            for token_id in elem.findall('token_anchor'):
                token_mention_id = token_id.get('t_id', 'nothing')
                event_dict[event_mention_id].append(token_mention_id)
        tag_set.add(elem.tag)
    return event_dict


def extract_plotLink(etreeRoot, event_dict):
    """提取事件对之间的关系, 返回事件关系词典 (key为事件对组成的元组即(e1,e2), value为关系)"""

    plot_dict = collections.defaultdict(list)
    for elem in etreeRoot.findall('Relations/'):
        if elem.tag == "PLOT_LINK":
            source_pl = elem.find('source').get('m_id', 'null')
            target_pl = elem.find('target').get('m_id', 'null')
            rel = elem.get('relType', 'null')
            if source_pl in event_dict:
                val1 = "_".join(event_dict[source_pl])
                if target_pl in event_dict:
                    val2 = "_".join(event_dict[target_pl])
                    plot_dict[(val1, val2)] = rel
    return plot_dict


def read_file(ecbstart_new, evaluate_file):
    """读取ECB文件与评估文件, 返回(事件词典, 事件关系词典, 评估数据)"""

    ecbstar = etree.parse(
        ecbstart_new, etree.XMLParser(remove_blank_text=True))
    ecbstar_root = ecbstar.getroot()
    ecbstar_root.getchildren()
    ecb_star_events = extract_event_CAT(ecbstar_root) # 提取所有事件 (事件m_id: 事件t_id列表(之后会被转换为t_id序列)) 
    ecbstar_events_plotLink = extract_plotLink(ecbstar_root, ecb_star_events) # 提取数据文件中所有事件关系
    evaluation_data = read_evaluation_file(evaluate_file) # 提取评估文件中所有事件关系
    return ecb_star_events, ecbstar_events_plotLink, evaluation_data


def make_corpus(ecbstartopic, evaluationtopic, datadict):
    """
    根据指定路径制作语料库
    
    :param ecbstartopic: 初始数据文件路径
    :param evaluationtopic: 评估文件路径
    :param datadict: 全局词典, key为文档名, value为[token词典, 事件词典, plotLink列表, evalData列表]
    """
    
    if os.path.isdir(ecbstartopic):
        if ecbstartopic[-1] != '/':
            ecbstartopic += '/'
        if evaluationtopic[-1] != '/':
            evaluationtopic += '/'
        
        file_count = 0
        for f in os.listdir(evaluationtopic):
            if f.endswith('plus.xml'):
                file_count += 1
                star_file = ecbstartopic + f + ".xml"
                evaluate_file = evaluationtopic + f
                ecb_star_events, ecbstar_events_plotLink, evaluation_data = read_file(star_file, evaluate_file)
                
                for key in ecb_star_events: # 将t_id列表转化为t_id序列
                    ecb_star_events[key] = '_'.join(ecb_star_events[key])

                all_token = all_tokens(star_file)
                datadict[star_file] = [all_token, ecb_star_events, ecbstar_events_plotLink, evaluation_data]


def extract_event_tokens(t_id_seq, all_tokens):
    """根据t_id序列获取对应的text序列"""

    tokens = []
    for t_id in t_id_seq.split('_'):
        text = all_tokens[int(t_id)-1][-1]
        tokens.append(text)
    return "_".join(tokens) 


def save_event_mention_txt(data_dict):
    """ 提取`ECI_ALL_Mention.txt` """

    # 使用dict中的key来记录text_seq (作用等于set)
    # 使用dict中的value来记录每个事件所属文件 (用于对异常事件进行定位)
    event_set_uncased = collections.defaultdict(list)

    for file in data_dict:
            all_tokens, event_dict = data_dict[file][:2]
            for m_id in event_dict:
                text_seq = extract_event_tokens(event_dict[m_id], all_tokens).lower()
                file_pre_name = file.split('/')[-1].split('.')[0][:-7]
                if file_pre_name not in event_set_uncased[text_seq]:
                    event_set_uncased[text_seq].append(file_pre_name)

    with open('./data/ECI_ALL_Mention.txt', 'w') as f:
        import re
        for event in event_set_uncased:
            f.write(f'{event}\t{event_set_uncased[event]}\n')
            if re.search('[^A-Za-z_]',event):
                print(event)
            

def main():
    version = 'v0.9'
    data_path = './data/EventStoryLine/'
    ECBstarTopic = data_path + 'annotated_data/' + version + '/'
    EvaluationTopic = data_path + 'evaluation_format/full_corpus/' + version + '/event_mentions_extended/'
    
    data_dict = {}
    for topic in os.listdir(ECBstarTopic): # 遍历数据集中的每一个topic
        if os.path.isdir(ECBstarTopic + topic):
            dir1, dir2 = ECBstarTopic + topic, EvaluationTopic+topic
            make_corpus(dir1, dir2, data_dict)

    # 保存读取到的数据
    with open('./data/document/document_ESC.pickle', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
