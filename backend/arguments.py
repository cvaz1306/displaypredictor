def extract_values_with_defaults(keys, argv=[]):
    r={}
    for key, default in keys:
        if key in argv:
            index=argv.index(key)
            value=argv[index+1]
            r[key]=value
        else:
            r[key]=value
    return r