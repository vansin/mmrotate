import os


prefix = 'work_dirs/headet/configs-ic19-rbb-rbb/'

groups = os.listdir(prefix)
mongo = 'headet0'

for i,group in enumerate(groups):

    print(f'group {i} of {len(groups)}: {group}')

    group_configs = os.listdir(f'{prefix}{group}')

    for j,config in enumerate(group_configs):
        
        print(f'config {j} of {len(group_configs)}: {config}')

        pths = os.listdir(f'{prefix}{group}/{config}')
        pths = [pth for pth in pths if pth.endswith('.pth')]

        sorted(pths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        os.system(f'python tools/test.py {prefix}{group}/{config}/{config}.py {prefix}{group}/{config}/{pths[-1]} --mongo {mongo}')
    