import os
import shutil
import sys
import time

def copy_tessdata(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")
    
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        
        if os.path.isdir(s):
            if item == 'scripts':
                # Copy contents of the scripts folder
                if not os.path.exists(d):
                    os.makedirs(d)
                    print(f"Created scripts directory: {d}")
                for script_item in os.listdir(s):
                    script_src = os.path.join(s, script_item)
                    script_dest = os.path.join(d, script_item)
                    if not os.path.exists(script_dest):
                        shutil.copy2(script_src, script_dest)
                        print(f"Copied {script_src} to {script_dest}")
                    else:
                        print(f"Skipped existing file: {script_dest}")
            else:
                if not os.path.exists(d):
                    shutil.copytree(s, d)
                    print(f"Copied directory {s} to {d}")
                else:
                    print(f"Skipped existing directory: {d}")
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)
                print(f"Copied {s} to {d}")
            else:
                print(f"Skipped existing file: {d}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python copy_tessdata.py <src_dir> <dest_dir>")
        sys.exit(1)

    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    copy_tessdata(src_dir, dest_dir)
    print("Copy operation completed.")
    time.sleep(10)  # Add a delay to keep the command prompt open for 10 seconds