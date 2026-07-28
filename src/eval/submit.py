"""Submit one zip to the BraTS Task-1 queue. CONSUMES A SLOT (2/day).

Usage: submit.py <zip> <name>
Recipe is handoff §5: keyring must be nulled headless; participants cannot list
submissions (403), so store+submit and read the id back from the return value.
"""
import os, sys

os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
import synapseclient
from synapseclient import File

ZIP = os.path.expanduser(sys.argv[1])
NAME = sys.argv[2]
PARENT = "syn75900400"
QUEUE = "9619537"

assert os.path.exists(ZIP), f"missing {ZIP}"
print(f"[submit] zip  = {ZIP} ({os.path.getsize(ZIP)} bytes)")
print(f"[submit] name = {NAME}")

tok = open(os.path.expanduser("~/brats2025/syn.tok")).read().strip()
syn = synapseclient.Synapse(silent=True, skip_checks=True)
syn.login(authToken=tok)

f = syn.store(File(ZIP, parent=PARENT))
print(f"[submit] stored entity {f.id}")

sub = syn.submit(evaluation=QUEUE, entity=f, name=NAME)
print(f"[submit] SUBMISSION_ID = {sub.id}")
print(f"[submit] name          = {sub.name}")
print("SUBMIT_COMPLETE")
