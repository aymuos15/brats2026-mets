import os, synapseclient
tok = open(os.path.expanduser("~/brats2025/syn.tok")).read().strip()
syn = synapseclient.Synapse()
syn.login(authToken=tok)
me = syn.getUserProfile()
print("LOGIN_OK user=", me.get("userName"), "id=", me.get("ownerId"))
ev = syn.getEvaluation("9619537")
print("EVAL", ev.name, "| status=", getattr(ev, "status", "?"))
# our recent submissions in this evaluation (confirm identity + count today)
try:
    subs = list(syn.getSubmissions("9619537", limit=10))
    mine = [s for s in subs if str(getattr(s, "userId", "")) == str(me.get("ownerId"))]
    print("VISIBLE_SUBS", len(subs), "MINE_recent", len(mine))
except Exception as e:
    print("subs_list_note", type(e).__name__, str(e)[:80])
print("CHECK_DONE")
