# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#获取边的权重(容量)
def getContact(fin):
    fin = open(fin)
    lines = fin.readlines()
    s1 = []
    s2 = []
    s3 = []
    for line in lines:
        line = line.strip()
        line = line.split()
        if float(line[4]) > 0:
            s1.append(line[0])
            s2.append(line[1])
            s3.append(line[4])
    #求概率最小值
    s3_min = min(s3)
    for i in range(len(s3)):
        s3[i] = float(s3[i])/float(s3_min)
    s = np.vstack((s1,s2))
    s = np.vstack((s,s3))
    s = s.T
    s = pd.DataFrame(s)
    return s
#获取s和t
#selection of extreme points
#1、选择三类特殊节点集合，由于我们这里只是知道接触概率，所以只产生一个节点集
#2、从三个集合中选取top 5%并去掉重复的点
#3、k标识离extreme相近的k个节点
def getSourceAndSink(data,num_k):
    data.columns = ['One','Two','Three']
    datalen = data['Two'].max()
    max_capacity = int(float(data['Three'].max())) + 1
    contact_density = []
    k_set = []
    for k in range(1,int(datalen)):
        cc = 0
        for i in range(1,k):
            for j in range(k+1,int(datalen)):
                 a = data[data.One == str(i)]
                 a = a[a.Two == str(j)]
                 if len(a) > 0 :
                        cc += float(a['Three'])
        cc = cc/(k*(int(datalen)-k))
        if cc != 0:
            k_set.append(k)
            contact_density.append(cc)
    s = np.vstack((k_set,contact_density)) 
    s = s.T
    s = pd.DataFrame(s)
    s.columns = ["One","Two"]
    #按照接触密度进行排序
    s = s.sort_values(by = ['Two'])
    #找出最小接触密度，残基对应当位于最小接触密度的不同侧
    #num_s残基对个数
    num_s = int(len(s)*0.05)
    #找到残基对
    #添加extreme Nodes
    #挑选距离最远的前5%的点
    #extreme_points = s.reset_index(drop=True)[0:num_s]
    extreme_points = s[0:num_s]
    extreme_source_points = []
    for i in extreme_points['One'].values:
        extreme_source_points.append(int(i))
    extreme_source_point = np.array(extreme_source_points)
    extreme_source_points = []
    #print("extreme_sinks_points:")
    for i in range(len(extreme_source_point)):
        extreme_source_points.append(extreme_source_point[i])
    #print(extreme_source_points)
    #找到source的extreme points
    #找到相离最远的点extreme_sinks_points
    extreme_sinks_points = []
    for k in range(len(extreme_source_points)):
        kk = extreme_source_points[k]
        tmp = 0
        sink_value = 0
        for i in range(len(data)):
            b = data[data.One == str(kk)]
            b = b[b.Two == str(i)]
            if len(b) > 0:
                sink_value = b['Three']
            if float(sink_value) > float(tmp):
                tmp = sink_value
                sink_value = tmp
                sink_points = i
        extreme_sinks_points.append(sink_points)
    #找到source点集，距离extreme最近的k个点
    #找到sink点集，距离extreme最近的k个点
    sink_sets = []
    source_sets = []
    #sinkToOther记录其他所有点到sink extreme的概率
    #sourceToother记录所有点到source extreme的概率
    sinkToOther = []
    sourceToOther = []
    #sink_index存放sink extreme节点
    #source_index存放source extreme节点
    sink_index = []
    source_index = []
    #only_index存放距离sink extreme接触的nodes节点
    only_index = []
    #得到距离 source extremes 的dataFrame
    for i in range(len(extreme_source_points)):
        kk = extreme_source_points[i]
        for j in range(1,int(datalen)+1):
            a = data[data.One == str(kk)]
            a = a[a.Two == str(j)]
            if len(a)>0:
                a  = float(a['Three'])
                only_index.append(int(j))
                source_index.append(kk)
                sourceToOther.append(int(a))
    only_index = np.array(only_index).T
    source_index = np.array(source_index).T
    sourceToOther = np.array(sourceToOther).T
    s_s_contact = np.vstack((only_index,source_index))
    s_s_contact = np.vstack((s_s_contact,sourceToOther))
    indexAndSource = pd.DataFrame(s_s_contact.T)
    indexAndSource.columns = ["index","kk","value"]
    print(indexAndSource.head())
    indexAndSource = indexAndSource.sort_values(by = ['kk','value'],ascending = [True,False])
    #从indexAndSource中找到每个source extreme点最近的k个点
    #先找到距离source最近的k个点集
    #将距离每个extreme前num_k点添加到sets中
    #index是距离第k个点的node节点，value是接触概率
    #tmp为每一次取出来的node
    for i in range(len(extreme_source_points)):
        kk = extreme_source_points[i]
        num_k_set = indexAndSource[indexAndSource.kk == int(kk)]
        #num_k_set索引没有改变
        num_k_set = num_k_set[0:num_k]
        #重新定义索引
        num_k_set = num_k_set.reset_index(drop = True)
        for i in range(0,len(num_k_set)):
            tmp = num_k_set.iloc[int(i),0]
            source_sets.append(tmp)
    print("source_sets:")
    print(source_sets)
    #再找到距离sink最近的k个点
    #将距离每个extreme前num_k点添加到sets中
    only_index_sink = []
    print("extreme_sinks_points")
    print(extreme_sinks_points)
    for i in range(len(extreme_sinks_points)):
        k = extreme_sinks_points[i]
        for j in range(1,int(datalen)+1):
            a = data[data.One == str(k)]
            a = a[a.Two == str(j)]
            if len(a)>0:
                a  = float(a['Three'])
                only_index_sink.append(int(j))
                sink_index.append(k)
                sinkToOther.append(int(a))
    #数组转置
    only_index_sink = np.array(only_index_sink).T
    sink_index = np.array(sink_index).T
    sinkToOther = np.array(sinkToOther).T
    ss_contact = np.vstack((only_index_sink,sink_index))
    ss_contact = np.vstack((ss_contact,sinkToOther))
    indexAndSink = pd.DataFrame(ss_contact.T)
    indexAndSink.columns = ["index","kk","value"]
    indexAndSink = indexAndSink.sort_values(by = ['kk','value'],ascending = [True,False])
    for i in range(len(extreme_sinks_points)):
        kk = extreme_sinks_points[i]
        num_kk_set = indexAndSink[indexAndSink.kk == int(kk)]
        num_kk_set = num_kk_set[0:num_k]
        num_kk_set = num_kk_set.reset_index(drop = True)
        for j in range(0,len(num_kk_set)):
            tmp = num_kk_set.iloc[int(j),0]
            sink_sets.append(tmp)
    print("sink_sets:")
    print(sink_sets)
    #将新的节点添加到data中   
    #添加新的节点s和t，以及source集合（source_sets）和sink集合(sink_sets)
    data_add = []
    for i in sink_sets:
        tmp = []
        tmp.append(float(i))
        tmp.append('t')
        tmp.append(float(max_capacity))
        data_add.append(tmp)
    for i in source_sets:
        tmp = []
        tmp.append('s')
        tmp.append(float(i))
        tmp.append(float(max_capacity))
        data_add.append(tmp)
    data_add = pd.DataFrame(data_add)
    data_add.columns = ["One","Two","Three"]
    p = [data,data_add]
    data = pd.concat(p, keys = ["One","Two","Three"])
    return data
#分割子图
def getGraph(data):
    #构建graph
    data = pd.DataFrame(data)
    data.columns = ['One','Two','Three']
    G = nx.DiGraph()
    for i in range(len(data["One"])):
        G.add_edge(data["One"][i],data["Two"][i],capacity = float(data["Three"][i]))
    #建立布局 pos = nx.spring_layout美化作用
    pos=nx.spring_layout(G)  
    #显示graph 
    edge_labels = nx.get_edge_attributes(G,'capacity') 
    nx.draw_networkx_nodes(G,pos) 
    nx.draw_networkx_labels(G,pos) 
    nx.draw_networkx_edges(G,pos) 
    nx.draw_networkx_edge_labels(G, pos,edge_labels) 
    plt.axis('on') 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.show() 
    #求最大流 
    #flow_value, flow_dict = nx.maximum_flow(G, '1', '492') 
    #print("最大流值: ",flow_value) 
    #print("最大流流经途径: ",flow_dict) 
    #图的分割
    #图的分割之前需要找到源s和汇点t
    cut_value,partition = nx.minimum_cut(G,'s','t')
    reachable,non_reachable = partition
    print("最大流：",cut_value)
    print("可到达的节点 ：",reachable)
    print("不可到达的节点：",non_reachable)
    #返回节点信息
    '''
    for i in range(0,len(reachable)-1):
        for j in range(i,len(reachable)):
            choiceData = data[data[0] == reachable[i] & data[1] == reachable[j]]
            print(choiceData)
    '''  
    return 0
if __name__ == "__main__":
    fin = "2xdpA.txt"
    #加载初始图的信息
    s = getContact(fin)
    k = 5
    #添加sink和source
    news = getSourceAndSink(s,k)
    #进行分割
    getGraph(news)