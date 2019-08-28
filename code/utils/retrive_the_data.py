import pandas as pd

#get the data length
def get_the_file_length(path):
	with open(path,"r") as f:
		lines = f.readlines()
		return len(lines)

def report_performance(fnames, repeats=5):
    paras = []
    perfs = []
    perf_count = 0
    val_perfs = []
    tes_perfs = []
    for fname in fnames:
        with open(fname) as fin:
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                line = line.replace('\'', '"')
                if '\t\t {"seq":' in line or '\t\t{"seq":' in line:
                    para = json.loads(line)
                    paras.append(para)
                    perf_count = 0
                    val_perfs = []
                    tes_perfs = []
                if 'Best Valid' in line:
                    val_per_str = line.replace('Best Valid performance: ', '')
                    val_per = json.loads(val_per_str)
                    tes_per_str = lines[ind + 1].replace('\'', '"').replace('\tBest Test performance: ', '')
                    # print(tes_per_str)
                    tes_per = json.loads(tes_per_str)
                    val_perfs.append(val_per)
                    tes_perfs.append(tes_per)
                    perf_count += 1
                    if perf_count == repeats:
                        perfs.append([val_perfs, tes_perfs])

    assert len(paras) == len(perfs), 'length of paras and perfs do not ' \
                                     'mathch %d VS %d' % (len(paras), len(perfs))
    # sort
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            if _check_para_order(paras[i], paras[j]):
                temp = paras[i]
                paras[i] = paras[j]
                paras[j] = temp

                temp = perfs[i]
                perfs[i] = perfs[j]
                perfs[j] = temp

    best_val_per = perfs[0][0][0]
    best_tes_per = perfs[0][1][0]
    for i in range(len(paras)):
        if paras[i]['alp'] > 5:
            continue
        # acl aw & pred
        if paras[i]['eps'] < 0.00075:
            continue
        # if paras[i]['bet'] > 1.05:
        #     continue
        if not (
                abs(paras[i]['bet'] - 0.001) < 1e-8 or
                abs(paras[i]['bet'] - 0.005) < 1e-8 or
                abs(paras[i]['bet'] - 0.01) < 1e-8 or
                abs(paras[i]['bet'] - 0.05) < 1e-8 or
                abs(paras[i]['bet'] - 0.1) < 1e-8 # or
                # abs(paras[i]['bet'] - 0.5) < 1e-8 # or
                #abs(paras[i]['bet'] - 1.0) < 1e-8
                ):
            continue

        # if paras[i]['eps'] < 0.00075:
        #     continue
        # # if paras[i]['bet'] < 0.001 or paras[i]['bet'] > 1.05:
        # if paras[i]['bet'] > 0.15:
        #     continue
        print(paras[i])
        cur_val_per = _get_mean_perf(perfs[i][0], repeats=repeats)
        print('Valid:', cur_val_per)
        cur_tes_per = _get_mean_perf(perfs[i][1], repeats=repeats)
        print('Test:', cur_tes_per)
        if cur_val_per['acc'] > best_val_per['acc']:
            best_val_per = cur_val_per
            best_tes_per = cur_tes_per

    print('Best valid perf:', best_val_per)
    print('Best test perf:', best_tes_per)


    # write out
    print('------------')
    for i in range(len(paras)):
        print(paras[i])
        print('Valid:', perfs[i][0])
        print('Test:', perfs[i][1])

def _check_para_order(par1, par2):
    if not par1['seq'] == par2['seq']:
        return par1['seq'] > par2['seq']
    elif not par1['unit'] == par2['unit']:
        return par1['unit'] > par2['unit']
    elif not par1['alp'] == par2['alp']:
        return par1['alp'] > par2['alp']
    elif not par1['bet'] == par2['bet']:
        return par1['bet'] > par2['bet']
    elif not par1['eps'] == par2['eps']:
        return par1['eps'] > par2['eps']
    else:
        return False

def _get_mean_perf(cur_perfs, repeats):
    mean_per = copy.copy(cur_perfs[0])
    for j in range(1, repeats):
        for metric in mean_per.keys():
            mean_per[metric] = mean_per[metric] + cur_perfs[j][metric]
    for metric in mean_per.keys():
        mean_per[metric] = mean_per[metric] / repeats
    return mean_per

def report_loss(fnames, ofname):
    tra_objs = []
    val_losses = []
    for fname in fnames:
        with open(fname) as fin:
            cur_objs = []
            cur_losses = []
            lines = fin.readlines()
            for ind, line in enumerate(lines):
                line = line.replace('\'', '"')
                if '----->>>>>' in line:
                    # value of objective function on training set
                    tra_obj = float(lines[ind + 1].split(' ')[2])
                    val_loss = float(lines[ind + 2].split('Val loss: ')[1])
                    cur_objs.append(tra_obj)
                    cur_losses.append(val_loss)
            tra_objs.append(cur_objs)
            val_losses.append(cur_losses)
    np.savetxt(ofname + 'tra_loss.csv', tra_objs, fmt='%.6f', delimiter=',')
    np.savetxt(ofname + 'val_loss.csv', val_losses, fmt='%.6f', delimiter=',')
#retrive the data from the file
def retrive_the_file(path):
	all_file = []
	sub_file = []
	all_voting_Str = 'the all folder voting precision is'
	lag_Str = 'the lag is'
	with open(path,"r") as f:
		lines = f.readlines()
		for i,line in enumerate(lines):
			if all_voting_Str.lower() in line.lower():
				file = []
				file.append(line.strip("\n").split(" ")[-1])
				print("the line is {} and the result is {}".format(line,line.strip("\n").split(" ")[-1]))
				for new_line in lines[i+1:i+10]:
					file.append(new_line.strip("\n").split(" ")[-1])
					print("the line is {} and the result is {}".format(line,line.strip("\n").split(" ")[-1]))
				sub_file.append(file)
				if lag_Str.lower() in lines[i+10].lower():
					for result in sub_file:
						result.append(lines[i+10].strip("\n").split(" ")[-1])
						result.append(lines[i+11].strip("\n").split(" ")[-1])
						result.append(lines[i+12].strip("\n").split(" ")[-1])
					all_file+=sub_file
					sub_file = []
	#print(all_file)
	file_dataframe = pd.DataFrame(all_file,columns=['all_voting','near_voting','far_voting','same_voting','reverse_voting','max_depth','learning_rate','gamma','min_child_weight','subsample','lag','train_date','test_date'])
	file_dataframe.to_csv("LMAHDY_h3_l30_xgbv6.csv",index=False)
				#for new_line in lines[i+1:i+13]:
					#if 

if __name__ == "__main__":
	path = "LMAHDY_h3_l30_xgbv6.txt"
	retrive_the_file(path)