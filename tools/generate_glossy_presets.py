import os
import json
import re

PRESETS_DIR = '/Users/wramkim/Documents/DEV/kolmar/presets'
TEMPLATE_FILE = 'glossy_test_260205_6.json'

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return list(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def main():
    # Load template
    with open(os.path.join(PRESETS_DIR, TEMPLATE_FILE), 'r', encoding='utf-8') as f:
        template_data = json.load(f)

    # Find source files
    source_files = {}
    for filename in os.listdir(PRESETS_DIR):
        if not filename.endswith('.json'):
            continue
        if filename == TEMPLATE_FILE:
            continue
        
        # Match pattern: [Season] [Type]_[Texture].json
        # Handle inconsistent casing if necessary, but assume standard format "Autumn Dark_Matte.json"
        # Some files might cover "Spring light_Matte.json" (lowercase l)
        
        # Regex to capture Season, Type, Texture
        match = re.match(r'^([A-Z][a-z]+)\s+([a-zA-Z]+)_(Matte|Satin)\.json$', filename)
        if match:
            season, type_name, texture = match.groups()
            key = f"{season} {type_name}"
            
            if key not in source_files:
                source_files[key] = {}
            
            source_files[key][texture] = filename

    # Process each color group
    created_count = 0
    for key, textures in source_files.items():
        # Prefer Satin, then Matte
        source_filename = textures.get('Satin', textures.get('Matte'))
        if not source_filename:
            continue
            
        print(f"Processing {key} using source: {source_filename}")
        
        with open(os.path.join(PRESETS_DIR, source_filename), 'r', encoding='utf-8') as f:
            source_data = json.load(f)
            
        # Create new data based on template
        new_data = template_data.copy()
        
        # Extract fields
        target_fields = ['LIP_COLOR_HEX', 'DEEP_COLOR', 'VALUE_WEIGHT']
        for field in target_fields:
            if field in source_data:
                new_data[field] = source_data[field]
            else:
                print(f"Warning: {field} not found in {source_filename}")

        # Update RGB if Hex changed
        if 'LIP_COLOR_HEX' in new_data:
            new_data['LIP_COLOR_RGB'] = hex_to_rgb(new_data['LIP_COLOR_HEX'])

        # Construct new filename
        # key is "Autumn Dark", output "Autumn Dark_Glossy.json"
        new_filename = f"{key}_Glossy.json"
        
        output_path = os.path.join(PRESETS_DIR, new_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            
        print(f"Created {new_filename}")
        created_count += 1

    print(f"Total {created_count} Glossy presets created.")

if __name__ == '__main__':
    main()
