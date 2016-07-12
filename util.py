def list_to_sql(list_obj):
    str_list_obj = [str(x) for x in list_obj]
    return '(' + ','.join(str_list_obj) + ')'