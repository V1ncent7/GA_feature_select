import json
import re
import os
def deal_file(path, s):
    #path = 'D:\\学习资料\\研二下\\test_0625\\GA\\json\\'
    path2 = 'D:\\学习资料\\研二下\\test_0625\\GA\\dnn_json\\'
    path3 = 'D:\\学习资料\\研二下\\test_0625\\GA\\result_json\\'
    files = os.listdir(path)
    for file in files:
        #print(file)
        if not os.path.isdir(file):
            f = open(path + "/" + file)
            result = []
            answer = []
            iter_f = iter(f)  # 用迭代器循环访问文件中的每一行
            features = []
            for line in iter_f:
                load_fea = json.loads(line)
                fea = load_fea['features']
                funcbb_fea = []
                for bb_fea in fea:
                    #print('origin fea:  ', bb_fea)
                    for item in s[::-1]:
                        bb_fea.remove(bb_fea[item])
                    funcbb_fea.append(bb_fea)
                features.append(funcbb_fea)
            f.close()

            f2 = open(path2 + "/" + file)
            f3 = open(path3 + "/" + file,'w')
            iter_f2 = iter(f2)
            index = 0
            for line in iter_f2:
                g_info = json.loads(line.strip())
                g_info['features'] = features[index]
                del g_info['func_features']
                index += 1
                json.dump(g_info, f3)
                f3.write('\n')
            f2.close()
            f3.close()
    #print("Run Model One Time!")




'''
path = 'D:\\学习资料\\研二下\\test_0625\\GA\\GeneticAlgorithmForFeatureSelection-master\\txt\\'
path2 = 'D:\\学习资料\\研二下\\test_0625\\GA\\GeneticAlgorithmForFeatureSelection-master\\result\\'
files= os.listdir(path)
s = []

for file in files:
    if not os.path.isdir(file):
        f = open(path+"/"+file)
        result= []
        answer = []
        iter_f=iter(f)      #用迭代器循环访问文件中的每一行
        #p = re.compile(r'\d+')
        f2 = open(path2+"/"+file,'w')
        f2.write("\"\",\"FuncCalls\",\"LogicInsts\",\"TransferInsts\","
                 "\"BasicBlocks\",\"Edges\",\"IncommingCalls\",\"Intrs\",\"between\"" + '\n')
        i = 0
        for line in iter_f:
            line = "\"" + str(i) + "\"," + str(line)

            f2.write(str(line))

        f.close()
        f2.close()
'''
