#coding=utf-8
import codecs

f_input = codecs.open('./data_path/labeled_data','r','utf-8')
f_output = codecs.open('./data_path/train_data','w','utf-8')
mapping_dict = {"实勘举报":1,"实勘预约":2,"实勘上传":3,"实勘图片":4}
weight_dict ={"实勘举报":1,"实勘预约":4,"实勘上传":5,"实勘图片":11}

def run():
    all_data = []
    for line in f_input:
        temp = str(line).strip('\n').replace('\r','').split('\t')
        if len(temp) != 6:
            continue
        pv = int(temp[0])
        query = temp[1]
        label = mapping_dict[temp[-1]]
        for i in range(pv * weight_dict[temp[-1]]):
            all_data.append((query,label))
    for (query,label) in all_data:
        f_output.write(query  + '\t' + str(label) + '\n')

if __name__ == "__main__":
    run()




