import pandas as pd
import os 
ans = None
for h in [1,3,5]:
    temp_ans = {"lag":[]}
    all_file = []
    for l in [5,10,20,30]:
        print(l)
        path = "all_l"+str(l)+"_h"+str(h)+"_v10_ex2_1718.txt"
        sub_file = []
        all_voting_Str = 'the all folder voting precision is'
        lag_Str = 'the lag is'
        with open(os.path.join(os.curdir,path),"r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                if all_voting_Str.lower() in line.lower():
                    file = []
                    file.append(float(line.strip("\n").split(" ")[-1]))
                    sub_file.append(file)
                    if lag_Str.lower() in lines[i+1].lower():
                        for result in sub_file:
                            result.append(lines[i+1].strip("\n").split(" ")[-1])
                            result.append(lines[i+3].strip("\n").split(" ")[-1])
                        all_file+=sub_file
                        sub_file = []
        temp_ans['lag'].append(str(l))
        temp_ans['lag'].append(str(l))
        temp_ans['lag'].append(str(l))
        temp_ans['lag'].append(str(l))
        temp_ans['lag'].append(str(l))
        temp_ans['lag'].append(str(l))
        
    for arr in all_file:
        if arr[2] not in temp_ans.keys():
            temp_ans[arr[2]] = [arr[0]]
        else:
            temp_ans[arr[2]].append(arr[0])
    print(temp_ans)
    temp_ans = pd.DataFrame(temp_ans)
    if ans is None:
        ans = temp_ans
    else:
        ans = pd.concat([ans,temp_ans], axis = 0, sort = False)

i = 0
true_ans = [None,None,None,None,None,None]
while i < len(ans):
    if true_ans[i%6] is None:
        true_ans[i%6]=pd.DataFrame(ans.iloc[i,:])
    else:
        true_ans[i%6] = pd.concat([true_ans[i%6],ans.iloc[i,:]],axis = 0, sort = False)
    i+=1

true_ans = pd.concat(true_ans,axis = 0, sort = False)
print(true_ans)
true_ans.to_csv("all_v10_ex2_res.csv")