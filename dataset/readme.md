1- for IoT_BDA _botnet analysis Dataset

#Dataset is excluded from this repository (We have not included dataset here due to file size contraints).
#Download Analysis dataset file (Analysis_Data.zip (Size: 25.35 GB) from IEEE DATA PORTAL, use following link 

link
https://ieee-dataport.org/open-access/iot-bda-botnet-analysis-dataset)

#Extract dataset files, into dataset directory (expected size for extracted files 78 GB) 
#Expected location (used in our pipeline):
../dataset/iot_bda_dataset/tasks/

#main artifacts used in our pipeline:

main files:
analysis_results.json
syscalls.json

optional files:
prog.log
machine.log

2- Dowload MITRE ATT&CK Enterprise stix dataset available in .json file

link: 
https://github.com/mitre-attack/attack-stix-data/blob/master/enterprise-attack/enterprise-attack.json

place it in dataset/mitre/*
we used in our pipeline: 
../dataset/mitre/enterprise-attack.json