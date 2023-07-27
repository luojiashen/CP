def color_print(type,f_code,b_code,s):
    print("\033[{};{};{}m{}\033[0m".format(type,f_code,b_code,s))
def c_print(s):
    color_print(0,33,40,s)
