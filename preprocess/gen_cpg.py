import os

path = 'ab_file\\'

dirs = os.listdir(path)

for d in dirs:
    if d == '.DS_Store':
        continue
    if os.path.exists(path + d + '\\cpg_a.txt') or os.path.exists(path + d + '\\cpg_b.txt'):
        continue

    # print(r'cd joern-cli && joern.bat --script ..\\locateFunc.sc --params "inputFile=.' + path + d + '\\a\\,outFile=.' + path + d + '\\cpg_a.txt"')
    os.system(r'cd joern && joern.bat --script ..\\locateFunc.sc --params "inputFile=..\\' + path + d + '\\a\\,outFile=..\\' + path + d + '\\cpg_a.txt"')

    os.system(r'cd joern && joern.bat --script ..\\locateFunc.sc --params "inputFile=..\\' + path + d + '\\b\\,outFile=..\\' + path + d + '\\cpg_b.txt"')

    os.system('python locate_and_align.py ' + path + d + '\\')

