import os
import json
import re

PRESETS_DIR = 'presets'

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return None
    return list(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def main():
    errors = []
    
    for filename in os.listdir(PRESETS_DIR):
        if not filename.endswith('.json'):
            continue
            
        path = os.path.join(PRESETS_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                errors.append(f"{filename}: JSON Decode Error")
                continue
                
        # Check HEX format
        hex_val = data.get('LIP_COLOR_HEX')
        if not hex_val:
            continue # Some presets might not have color?
            
        if not re.match(r'^#[0-9a-fA-F]{6}$', hex_val):
            errors.append(f"{filename}: Invalid HEX format '{hex_val}'")
            continue
            
        # Check RGB match
        rgb_val = data.get('LIP_COLOR_RGB')
        if rgb_val:
            calc_rgb = hex_to_rgb(hex_val)
            if calc_rgb != rgb_val:
                errors.append(f"{filename}: Mismatch! HEX {hex_val} -> {calc_rgb}, but file has {rgb_val}")
        else:
            # If RGB is missing, it's not strictly an error but good to know
            pass

    if errors:
        print("Found errors:")
        for e in errors:
            print(e)
    else:
        print("All presets are valid.")

if __name__ == '__main__':
    main()
