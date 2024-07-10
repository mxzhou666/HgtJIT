import csv
import os
import shutil
import sys
csv.field_size_limit(5000000)
with open('dataset/C_dataset.csv', newline='', encoding='utf-8') as csvfile:
	reader = csv.reader(csvfile)
	data_list = list()
	count = 0
	for row in reader:
		commit_ID = row[0]
		if commit_ID == 'commit_id':
			continue
		# create a folder to store pre- and post-patch version files
		if not os.path.exists('dataset/ab_file/'):
			os.mkdir('dataset/ab_file/')
		# if not os.path.exists('repo/'):
		# 	os.mkdir('repo/')

		if not os.path.exists('dataset/ab_file/'+commit_ID):
			# os.system('mkdir ab_file/'+commit_ID)
			os.system('mkdir "dataset/ab_file/' + commit_ID + '"')
			os.system('mkdir "dataset/ab_file/' + commit_ID + '/a"')
			os.system('mkdir "dataset/ab_file/' + commit_ID + '/b"')
		print('存储:' + commit_ID)
		filename = row[4]
		i = filename.rfind('/')
		if i != 0 :
			filename = filename[i + 1:]

		file_path_a = 'dataset/ab_file/' + commit_ID + '/a/' + filename
		file_path_b = 'dataset/ab_file/' + commit_ID + '/b/' + filename
		with open(file_path_a, 'w', encoding='utf-8') as file:
			file.write(row[6])
		with open(file_path_b, 'w', encoding='utf-8') as file:
			file.write(row[7])




