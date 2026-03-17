@echo off
cd /d %~dp0
python -m PyInstaller --noconfirm --clean --windowed --name PencilSketchApp ^
  --add-data "pencil_sketch_app;pencil_sketch_app" ^
  --hidden-import PIL._tkinter_finder ^
  main.py
pause
