from PyInstaller import __main__ as m

DIST = "--distpath"
NAME = "--name"
UPX = "--upx-dir"
LIB_PATH = "--paths"
EXCLUDES = "--exclude-module"
PATH = "--path"

py_script = "trainer_gui.py"
exe_name = "DATASETMANAGER"

addtl_args = ["--onefile",
              "--windowed", 
              "--clean"]

venv_path = "~/Documents/Code/venv/dataset_generator/Lib/site-packages"

dist_path = "."

upx_path = "~/Documents/Code/Git/upx-4.0.2-win64"

search_pathes = [".", 
                 "~/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/*"]

pyinstaller_mods = ["PyInstaller",
                    "altgraph",
                    "pywin32-ctypes",
                    "pyinstaller-hooks-contrib",
                    "pefile"]
other_mod_excludes = []

args = [py_script]
for a in addtl_args:
    args.append(a)

args.append(DIST)
args.append(dist_path)
args.append(NAME)
args.append(exe_name)
args.append(UPX)
args.append(upx_path)
args.append(PATH)
args.append(venv_path)

for p in search_pathes:
    args.append(LIB_PATH)
    args.append(p)

for mod in pyinstaller_mods:
    args.append(EXCLUDES)
    args.append(mod)

for mod in other_mod_excludes:
    args.append(EXCLUDES)
    args.append(mod)

m.run(args)
