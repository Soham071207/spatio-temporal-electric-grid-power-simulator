import os
import zipfile

def zip_project():
    zipf = zipfile.ZipFile('spain_digital_twin_source.zip', 'w', zipfile.ZIP_DEFLATED)
    
    # Exclude massive folders, models, node_modules, and git
    exclusions = ['node_modules', 'venv', '.git', '__pycache__', 'data', '.gemini', 'forecast_results', '.vscode']
    
    for root, dirs, files in os.walk('.'):
        # Modify dirs in-place to prune the walk
        dirs[:] = [d for d in dirs if d not in exclusions and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.zip') or file.endswith('.pyc') or file.endswith('.log'):
                continue
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, '.')
            zipf.write(filepath, arcname)
            print(f"Added: {arcname}")
            
    zipf.close()
    print("Successfully zipped project to spain_digital_twin_source.zip")

if __name__ == "__main__":
    zip_project()
