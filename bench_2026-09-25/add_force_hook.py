import io
p = "/home/shavin/llm-cpu-compiler/pytorch/torch/_inductor/kernel/mm_common.py"
s = io.open(p, encoding="utf-8").read()
old = """    int_max = torch.iinfo(torch.int32).max
    try:
        size = node.get_layout().storage_size()"""
new = """    import os

    if os.environ.get("INDUCTOR_FORCE_I64_OFFSETS") == "1":
        # Test hook: exercise the int64 offset path on an operand small enough
        # to also run through the int32 path, so the two can be compared.
        return True
    int_max = torch.iinfo(torch.int32).max
    try:
        size = node.get_layout().storage_size()"""
assert s.count(old) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("force hook added")
