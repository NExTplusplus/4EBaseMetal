def deal_with_minor_OI():
    for column_name in column_list:
        value_list=[]
        value_dict={}
        for item in data['datetime']:
            value = list(data[data['datetime']==item][column_name])[0]
            value_list.append((item,value))
        value_list = sorted(value_list,key=lambda x:x[1],reverse=False)
        for i in range(20):
            d = datetime.strptime(value_list[i][0], '%Y-%m-%d')
            year = d.year
            month = d.month
            start_time = str(year)+'-'+str(month)+'-'+'01'
            end_time = str(year)+'-'+str(month)+'-'+'31'
            month_value_list = list(data[(data['datetime']>=start_time)&(data['datetime']<=end_time)][column_name])
            data[data['datetime']==value_list[i][0]][column_name]=np.mean(month_value_list)
    return data