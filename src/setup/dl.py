import synapseclient, os
syn = synapseclient.Synapse()
syn.login(authToken=os.environ['SYN_TOKEN'])
dest = os.path.expanduser('~/brats2025/raw_zips')
for sid in ['syn64919665','syn64919141','syn65888166']:
    e = syn.get(sid, downloadLocation=dest)
    print('GOT', sid, '->', e.name, flush=True)
print('DOWNLOAD COMPLETE', flush=True)
