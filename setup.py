from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {'packages': [], 'excludes': []}

base = 'Console'

executables = [
    Executable('main.py', base=base, target_name = 'crocrodile')
]

setup(name='Crocrodile',
      version = '0.4',
      description = 'Simple chess engine',
      options = {'build_exe': build_options},
      executables = executables)
