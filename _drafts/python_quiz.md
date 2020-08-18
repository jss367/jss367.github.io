






def quiz():
    try:
        return True
    finally:
        return False


s = {0}
for i in s:
    s.add(i + 1)
    s.remove(i)

    